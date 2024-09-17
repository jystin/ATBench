from detectron2.utils.file_io import PathManager
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from atmodel.language.LangEncoder import build_tokenizer

from fastai.vision import *
import string



CAPTIONING_DATASETS = {
    "vizwiz_captioning_train": (
        "captioning_datasets/vizwiz/train",
        "captioning_datasets/vizwiz/annotations/train.json",
    ),
    "vizwiz_captioning_val": (
        "captioning_datasets/vizwiz/val",
        "captioning_datasets/vizwiz/annotations/val.json",
    ),
    "vizwiz_captioning_test": (
        "captioning_datasets/vizwiz/test",
        "captioning_datasets/vizwiz/annotations/test.json",
    ),
}


class CaptioningDataset(Dataset):
    "`PanoSegDataset` read segmentation data."
    def __init__(self,
                 cfg,
                 dataset_name,
                 is_train=True):
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.image_format = cfg['INPUT']['FORMAT']
        self.max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        self.tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])

        self.info = self.load_dataset(self.dataset_name)
        self.tfm_gens = self.build_transform_gen(cfg, is_train)
        self.meta = self.get_metadata()
        self.set_metadata(dataset_name, evaluator_type=cfg['INPUT']['EVALUATOR_TYPE'])

    def get_metadata(self):
        meta = {}
        return meta

    def load_dataset(self, dataset_name):
        root = os.getenv("DATASET", "datasets")
        image_root, annotation_file = CAPTIONING_DATASETS[dataset_name]

        with PathManager.open(os.path.join(root, annotation_file), "r") as f:
            anno = json.load(f)

        split = dataset_name.split("_")[-1]

        images = anno["images"]
        captions = anno["annotations"] if anno.get("annotations") else None


        dataset_info = {
                "split": split,
                "image_root": os.path.join(root, image_root),
                "annotation_file": os.path.join(root, annotation_file),
                "images": images,
                "captions": captions}

        return dataset_info

    def set_metadata(self, dataset_name, evaluator_type):
        MetadataCatalog.get(dataset_name).set(
            image_root=self.info['image_root'],
            gt_json=self.info['annotation_file'],
            evaluator_type=evaluator_type,
            **self.meta,
        )

    def __len__(self):
        return len(self.info["images"])

    def __getitem__(self, idx):
        image = self.info["images"][idx]
        image_name = image["file_name"]
        image_id = image["id"]
        file_name = os.path.join(self.info["image_root"], image_name)

        dataset_dict = {
            "file_name": file_name,
            "image_id": image_id,
        }

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        question = ["what does the image describe?".lower()]
        dataset_dict['question'] = question
        question_token = self.tokenizer(
            question, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )
        dataset_dict['question_tokens'] = {"input_ids": question_token["input_ids"], "attention_mask": question_token["attention_mask"]}

        if not self.is_train:
            return dataset_dict

        transtab = str.maketrans({key: None for key in string.punctuation})
        captions = []
        for i in range(idx*5, idx*5+5):
            cap_dict = self.info["captions"][i]
            assert cap_dict["image_id"] == image_id, "caption does not match image"
            cap = cap_dict["caption"]
            cap.replace("\n", ", ")
            cap = cap.translate(transtab)
            captions.append(cap)

        dataset_dict['answers'] = captions
        answer_token = self.tokenizer(
            captions, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )
        dataset_dict['answer_tokens'] = {"input_ids": answer_token["input_ids"], "attention_mask": answer_token["attention_mask"]}


        return dataset_dict

    def build_transform_gen(self, cfg, is_train):
        """
        Create a list of default :class:`Augmentation` from config.
        Now it includes resizing and flipping.
        Returns:
            list[Augmentation]
        """
        # The scope of vlp dataset may not need any augmentation.
        cfg_input = cfg['INPUT']
        image_size = cfg_input['IMAGE_SIZE']
        augmentation = []

        augmentation.extend([
            T.Resize((image_size[0], image_size[1])),
        ])

        return augmentation



