# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.


import os
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data
import torch.utils.data as torchdata
from torch.utils.data import ConcatDataset

import detectron2.utils.comm as comm
from detectron2.data.build import (
    build_batch_data_loader,
    load_proposals_into_dataset,
    trivial_batch_collator,
)
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
from fvcore.common.config import CfgNode

from .dataset_mappers import (
    OCRDataset,
    PanoSegDataset,
    CaptioningDataset,
    VQADataset,
    DepthDataset,
)
from .evaluation import (InstanceSegEvaluator,
                         SemSegEvaluator,
                         CaptioningEvaluator,
                         COCOPanopticEvaluator,
                         OcrEvaluator,
                         DepthEvaluator,
                         VQAEvaluator,
                         )
from atmodel.utils import configurable
from utils.distributed import get_world_size


class JointLoader(torchdata.IterableDataset):
    def __init__(self, loaders, key_dataset):
        # loaders :  dict of torchdata.DataLoader ["seg": :torchdata.DataLoader, "vlp"::torchdata.DataLoader, "ocr"::torchdata.DataLoader]
        dataset_names = []
        for key, loader in loaders.items():
            name = "{}".format(key.split('_')[0])
            setattr(self, name, loader)
            dataset_names += [name]
        self.dataset_names = dataset_names
        self.key_dataset = key_dataset

    def __iter__(self):
        for batch in zip(*[getattr(self, name) for name in self.dataset_names]):
            yield {key: batch[i] for i, key in enumerate(self.dataset_names)}

    def __len__(self):
        return len(getattr(self, self.key_dataset))


def filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if isinstance(ann, list):
                for instance in ann:
                    if instance.get("iscrowd", 0) == 0:
                        return True
            else:
                if ann.get("iscrowd", 0) == 0:
                    return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def get_detection_dataset_dicts(
        dataset_names, filter_empty=True, proposal_files=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)

    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    # instance seg?
    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names)

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))

    # dict_keys(['file_name', 'image_id', 'captions', 'grounding_info', 'pan_seg_file_name', 'sem_seg_file_name', 'segments_info'])
    return dataset_dicts


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=None,
    )
    if mapper is None:
        mapper_cfg = CfgNode({'INPUT': cfg['INPUT'], 'MODEL': cfg['MODEL'], 'DATASETS': cfg['DATASETS']})
        mapper = DatasetMapper(mapper_cfg, False)
    assert cfg['TEST'][
               'BATCH_SIZE_TOTAL'] % get_world_size() == 0, "Evaluation total batchsize is not divisible by gpu number"
    batch_size = cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size()

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg['DATALOADER']['NUM_WORKERS'],
        "sampler": InferenceSampler(len(dataset)),
        "batch_size": batch_size,
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
        dataset: Union[List[Any], torchdata.Dataset],
        *,
        mapper: Callable[[Dict[str, Any]], Any],
        sampler: Optional[torchdata.Sampler] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )

def build_test_loader(
        dataset: Union[List[Any], torchdata.Dataset],
        *,
        sampler: Optional[torchdata.Sampler] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
):

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,)



def _train_loader_from_config(cfg, dataset_name, mapper, *, dataset=None, sampler=None):
    cfg_datasets = cfg['DATASETS']
    cfg_dataloader = cfg['DATALOADER']

    # print(dataset_name)  # coco_2017_train_panoptic_filtall_with_sem_seg_caption_grounding
    if dataset is None:
        # dataset is a datainfo dict like the return in register file
        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg_dataloader['FILTER_EMPTY_ANNOTATIONS'],
            proposal_files=cfg_datasets['PROPOSAL_FILES_TRAIN'] if cfg_dataloader['LOAD_PROPOSALS'] else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg_dataloader['SAMPLER_TRAIN']
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        sampler = TrainingSampler(len(dataset))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg['TRAIN']['BATCH_SIZE_TOTAL'],
        "aspect_ratio_grouping": cfg_dataloader['ASPECT_RATIO_GROUPING'],
        "num_workers": cfg_dataloader['NUM_WORKERS'],
    }


@configurable(from_config=_train_loader_from_config) 
def build_detection_train_loader(
        dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )

def build_train_loader(
        dataset, sampler=None, total_batch_size=1, aspect_ratio_grouping=True, num_workers=0,
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


def get_config_from_name(cfg, dataset_name):
    # adjust config according to dataset
    if 'ade20k_panoptic' in dataset_name:
        cfg.update(cfg['ADE20K'])
        return cfg
    elif 'vizwiz_captioning' in dataset_name:
        cfg.update(cfg['VIZWIZ_CAPTIONING'])
        return cfg
    elif 'vizwiz_vqa' in dataset_name:
        cfg.update(cfg['VIZWIZ_VQA'])
        return cfg
    elif 'nyuv2_depth' in dataset_name:
        cfg.update(cfg['NYU_V2'])
        return cfg
    elif "ocr" in dataset_name:
        cfg.update(cfg["OCR"])
        return cfg
    else:
        assert False, "dataset not support."


def build_eval_dataloader(cfg, ):
    dataloaders = []
    for task_name in cfg['DATASETS']['TEST']:
        cfg = get_config_from_name(cfg, task_name)
        # adjust mapper according to dataset
        if task_name in ['ade20k_panoptic_val']:
            dataset = create_dataset(cfg, PanoSegDataset, is_train=False)
            dataloaders += [build_test_loader(dataset, batch_size=cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size(),
                                                    num_workers=cfg['DATALOADER']['NUM_WORKERS'])]
        elif task_name in ["vizwiz_captioning_val"]:
            dataset = create_dataset(cfg, CaptioningDataset, is_train=False)
            dataloaders += [build_test_loader(dataset, batch_size=cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size(),
                                                    num_workers=cfg['DATALOADER']['NUM_WORKERS'])]
        elif task_name in ["vqav2_val", "vizwiz_vqa_val"]:
            dataset = create_dataset(cfg, VQADataset, is_train=False)
            dataloaders += [build_test_loader(dataset, batch_size=cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size(),
                                                    num_workers=cfg['DATALOADER']['NUM_WORKERS'])]
        elif task_name in ["ocr_val"]:
            dataset = create_dataset(cfg, OCRDataset, is_train=False, task_name=task_name)
            dataloaders += [build_test_loader(dataset, batch_size=cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size(),
                                                    num_workers=cfg['DATALOADER']['NUM_WORKERS'])]
        elif task_name in ["nyuv2_depth_val", "kitti_depth_val"]:
            dataset = create_dataset(cfg, DepthDataset, is_train=False)
            dataloaders += [build_test_loader(dataset, batch_size=cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size(),
                                                    num_workers=cfg['DATALOADER']['NUM_WORKERS'])]
        else:
            mapper = None
            dataloaders += [build_detection_test_loader(cfg, task_name, mapper=mapper)]

    return dataloaders

class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets, task_name=None, evaluator_type=None):
        super(MyConcatDataset, self).__init__(datasets)
        self.meta = self.get_metadata()
        self.set_metadata(task_name, evaluator_type=evaluator_type)

    def __getattr__(self, k):
        return getattr(self.datasets[0], k)

    def get_metadata(self):
        meta = {}
        return meta

    def set_metadata(self, dataset_name, evaluator_type):
        MetadataCatalog.get(dataset_name).set(
            evaluator_type=evaluator_type,
            **self.meta,
        )

def create_dataset(cfg, dataset_mapper, is_train=True, task_name=None):
    if is_train:
        if cfg["MULTI_DATASETS"]:
            path_list = cfg['TRAIN_DATASET_NAME']
            datasets = [dataset_mapper(cfg, dataset_name, is_train=is_train) for dataset_name in path_list]
            dataset = MyConcatDataset(datasets, task_name=task_name, evaluator_type=cfg['INPUT']['EVALUATOR_TYPE'])
        else:
            dataset_name = cfg['TRAIN_DATASET_NAME']
            dataset = dataset_mapper(cfg, dataset_name, is_train=is_train)
    else:
        if cfg["MULTI_DATASETS"]:
            path_list = cfg['TEST_DATASET_NAME']
            datasets = [dataset_mapper(cfg, dataset_name, is_train=is_train) for dataset_name in path_list]
            dataset = MyConcatDataset(datasets, task_name=task_name, evaluator_type=cfg['INPUT']['EVALUATOR_TYPE'])
        else:
            dataset_name = cfg['TEST_DATASET_NAME']
            dataset = dataset_mapper(cfg, dataset_name, is_train=is_train)
    return dataset

def build_train_dataloader(cfg):
    task_names = cfg['DATASETS']['TRAIN']

    loaders = {}
    for task_name in task_names:
        # update the config file
        cfg = get_config_from_name(cfg, task_name)
        mapper_name = cfg['INPUT']['DATASET_MAPPER_NAME']  # e.g.: "panoptic"
        # print(mapper_name)

        # panoptic segmentation
        if mapper_name == "panoptic":
            dataset = create_dataset(cfg, PanoSegDataset)
            loaders['seg'] = build_train_loader(dataset, total_batch_size=cfg['TRAIN']['BATCH_SIZE_TOTAL'],
                                                         aspect_ratio_grouping=cfg['DATALOADER']['ASPECT_RATIO_GROUPING'],
                                                         num_workers=cfg['DATALOADER']['NUM_WORKERS'])
        # depth
        elif mapper_name == "depth":
            dataset = create_dataset(cfg, DepthDataset)
            loaders['depth'] = build_train_loader(dataset, total_batch_size=cfg['TRAIN']['BATCH_SIZE_TOTAL'],
                                                         aspect_ratio_grouping=cfg['DATALOADER']['ASPECT_RATIO_GROUPING'],
                                                         num_workers=cfg['DATALOADER']['NUM_WORKERS'])
        # captioning
        elif mapper_name == "cap":
            dataset = create_dataset(cfg, CaptioningDataset)
            loaders['cap'] = build_train_loader(dataset, total_batch_size=cfg['TRAIN']['BATCH_SIZE_TOTAL'],
                                                         aspect_ratio_grouping=cfg['DATALOADER']['ASPECT_RATIO_GROUPING'],
                                                         num_workers=cfg['DATALOADER']['NUM_WORKERS'])
        # vqa
        elif mapper_name == "vqa":
            dataset = create_dataset(cfg, VQADataset)
            loaders['vqa'] = build_train_loader(dataset, total_batch_size=cfg['TRAIN']['BATCH_SIZE_TOTAL'],
                                                         aspect_ratio_grouping=cfg['DATALOADER']['ASPECT_RATIO_GROUPING'],
                                                         num_workers=cfg['DATALOADER']['NUM_WORKERS'])
        # ocr
        elif mapper_name == "ocr":
            dataset = create_dataset(cfg, OCRDataset, task_name=task_name)
            loaders['ocr'] = build_train_loader(dataset, total_batch_size=cfg['TRAIN']['BATCH_SIZE_TOTAL'],
                                                         aspect_ratio_grouping=cfg['DATALOADER']['ASPECT_RATIO_GROUPING'],
                                                         num_workers=cfg['DATALOADER']['NUM_WORKERS'])
        else:
            raise NotImplementedError("mapper_name {} not supported.".format(mapper_name))


    if len(loaders) == 1 and not cfg['LOADER'].get('JOINT', False):
        return list(loaders.values())[0]
    else:
        return JointLoader(loaders, key_dataset=cfg['LOADER'].get('KEY_DATASET', 'seg'))


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg["SAVE_DIR"], "inference")
    evaluator_list = []


    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # print(evaluator_type)
    # exit()

    # semantic segmentation
    if evaluator_type in ["sem_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    # instance segmentation
    if evaluator_type == "coco":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

    cfg_model_decoder_test = cfg["MODEL"]["DECODER"]["TEST"]
    # panoptic segmentation
    if evaluator_type in [
        "panoptic_seg",
    ]:
        if cfg_model_decoder_test["PANOPTIC_ON"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    # COCO
    if (evaluator_type == "coco_panoptic_seg" and cfg_model_decoder_test[
        "INSTANCE_ON"]) or evaluator_type == "object365_od":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if (evaluator_type == "coco_panoptic_seg" and cfg_model_decoder_test[
        "SEMANTIC_ON"]) or evaluator_type == "coco_sem_seg":
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))

    # ADE20K
    if evaluator_type == "ade20k_panoptic_seg" and cfg_model_decoder_test["INSTANCE_ON"]:
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "captioning":
        evaluator_list.append(
            CaptioningEvaluator(dataset_name, output_dir=output_folder, gt_json=MetadataCatalog.get(dataset_name).gt_json))
    if evaluator_type == "vqa":
        evaluator_list.append(VQAEvaluator(dataset_name, output_dir=output_folder,
                                           anno_json=MetadataCatalog.get(dataset_name).anno_json,
                                           question_json=MetadataCatalog.get(dataset_name).question_json))
    if evaluator_type == "ocr":
        evaluator_list.append(OcrEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "depth":
        evaluator_list.append(DepthEvaluator(dataset_name, output_dir=output_folder))

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]

    return DatasetEvaluators(evaluator_list)