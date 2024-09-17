from detectron2.utils.file_io import PathManager
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from atmodel.language.LangEncoder import build_tokenizer
from detectron2.projects.point_rend import ColorAugSSDTransform

from fastai.vision import *


DEPTH_DATASETS = {
    "nyuv2_depth_train": (
        "depth_datasets/nyuv2/images",
        "depth_datasets/nyuv2/raw_depths",
        "depth_datasets/nyuv2/train.txt",
    ),
    "nyuv2_depth_val": (
        "depth_datasets/nyuv2/images",
        "depth_datasets/nyuv2/raw_depths",
        "depth_datasets/nyuv2/val.txt",
    ),
}


class DepthDataset(Dataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 is_train=True):
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.image_format = cfg['INPUT']['FORMAT']
        self.max_depth = cfg['INPUT']["MAX_DEPTH"]
        self.max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        self.tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])

        self.info = self.load_dataset(self.dataset_name)
        self.tfm_gens = self.build_transform_gen(cfg, is_train)
        self.meta = self.get_metadata()
        self.set_metadata(dataset_name, evaluator_type=cfg['INPUT']['EVALUATOR_TYPE'])

    def get_metadata(self):
        meta = {}
        return meta

    def get_vocab(self, path):
        with open(path, 'r') as fd:
            pre_vocabs = json.load(fd)
        return pre_vocabs["answer"]

    def load_dataset(self, dataset_name):
        root = os.getenv("DATASET", "datasets")
        image_root, depth_root, split_file = DEPTH_DATASETS[dataset_name]

        with PathManager.open(os.path.join(root, split_file), "r") as f:
            images = f.readlines()

        dataset_info = {
                "image_root": os.path.join(root, image_root),
                "depth_root": os.path.join(root, depth_root),
                "split_file": os.path.join(root, split_file),
                "images": images}

        return dataset_info

    def set_metadata(self, dataset_name, evaluator_type):
        MetadataCatalog.get(dataset_name).set(
            image_root=self.info['image_root'],
            depth_root=self.info['depth_root'],
            split_file=self.info['split_file'],
            evaluator_type=evaluator_type,
            **self.meta,
        )

    def __len__(self):
        return len(self.info["images"])

    def __getitem__(self, idx):
        img_num = self.info["images"][idx]
        img_num = img_num.strip()
        image_name = str(img_num) + ".jpg"
        depth_name = str(img_num) + ".png"
        image_id = int(img_num)
        image_file = os.path.join(self.info["image_root"], image_name)
        depth_file = os.path.join(self.info["depth_root"], depth_name)

        dataset_dict = {
            "file_name": image_file,
            "depth_file": depth_file,
            "image_id": image_id,
        }

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        depth_gt = utils.read_image(dataset_dict.pop("depth_file")).astype("double")
        depth_gt = transforms.apply_segmentation(depth_gt)

        depth_gt = depth_gt / 1000.0

        dataset_dict["depth_map"] = torch.as_tensor(np.ascontiguousarray(depth_gt))
        dataset_dict["depth_max_depth"] = self.max_depth


        question = ["what is the depth estimation of this image?".lower()]
        dataset_dict['question'] = question

        question_token = self.tokenizer(
            question, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )
        dataset_dict['question_tokens'] = {"input_ids": question_token["input_ids"],
                                           "attention_mask": question_token["attention_mask"]}
        return dataset_dict



    def build_transform_gen(self, cfg, is_train):
        """
        Create a list of default :class:`Augmentation` from config.
        Now it includes resizing and flipping.
        Returns:
            list[Augmentation]
        """
        cfg_input = cfg['INPUT']
        image_size = cfg_input['IMAGE_SIZE']
        min_scale = cfg_input['MIN_SCALE']
        max_scale = cfg_input['MAX_SCALE']
        angle = cfg_input['ANGLE']  
        crop_size = cfg_input['CROP_SIZE']

        augmentation = []
        if is_train:
            if cfg_input['RANDOM_FLIP'] != "none":
                augmentation.append(
                    T.RandomFlip(
                        horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                        vertical=cfg_input['RANDOM_FLIP'] == "vertical",
                    )
                )
            augmentation.extend([
                # T.RandomRotation(angle=angle),
                T.ResizeScale(
                    min_scale=min_scale, max_scale=max_scale, target_height=image_size[0], target_width=image_size[1]
                ),
                T.FixedSizeCrop(crop_size=(crop_size)),
            ])

            if cfg_input['COLOR_AUG_SSD']:
                augmentation.append(ColorAugSSDTransform(img_format=cfg_input['FORMAT']))

        else:
            augmentation.append(T.Resize((image_size[0], image_size[1]), interp=PIL.Image.BICUBIC))

        return augmentation
