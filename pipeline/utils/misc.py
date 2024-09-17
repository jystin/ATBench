import logging
import torch

logger = logging.getLogger(__name__)


# HACK for evalution 
def hook_metadata(metadata, name):
    if name == 'cityscapes_fine_sem_seg_val':
        metadata.__setattr__("keep_sem_bgd", True)
    return metadata

# HACK for evalution 
def hook_switcher(model, name):
    mappings = {}
    if name in ['ade20k_full_sem_seg_val']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False}
    elif name in ['ade20k_panoptic_val', 'bdd10k_40_panoptic_val']:
        mappings = {'SEMANTIC_ON': False, 'INSTANCE_ON': False, 'PANOPTIC_ON': True}
    else:
        if name not in ["nyuv2_depth_val", "kitti_depth_val", "coco_ocr_val", "Syn90k_val", "CUTE80_val", "IC13_val", "IC15_val", "IIIT5K_val", "SVT_val", "SVTP_val",
                        "coco_captioning_val", "vizwiz_captioning_val", "vizwiz_vqa_val", "vqav2_val", "ocr_val"]:
            assert False, "dataset switcher is not defined"
    for key, value in mappings.items():
        if key == 'SEMANTIC_ON':
            model.model.semantic_on = value
        if key == 'INSTANCE_ON':
            model.model.instance_on = value
        if key == 'PANOPTIC_ON':
            model.model.panoptic_on = value