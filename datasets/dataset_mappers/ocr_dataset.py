import logging
import cv2
import lmdb
import six
from fastai.vision import *
from torchvision import transforms
from atmodel.language.LangEncoder import build_tokenizer

from ..utils.transforms import CVColorJitter, CVDeterioration, CVGeometry
from ..utils.utils import CharsetMapper, onehot
from PIL import Image

OCR_DATASETS = {
    "mj_train": (
        "atmodel_data/ocr_datasets/training/MJ/MJ_train",
    ),
    "mj_val": (
        "atmodel_data/ocr_datasets/training/MJ/MJ_valid",
    ),
    "mj_test": (
        "atmodel_data/ocr_datasets/training/MJ/MJ_test",
    ),
    "st_train": (
        "atmodel_data/ocr_datasets/training/ST",
    ),
    "iiit50_3000": (
        "atmodel_data/ocr_datasets/evaluation/IIIT5k_3000",
    ),
    "svt": (
        "atmodel_data/ocr_datasets/evaluation/SVT",
    ),
    "svtp": (
        "atmodel_data/ocr_datasets/evaluation/SVTP",
    ),
    "ic13": (
        "atmodel_data/ocr_datasets/evaluation/IC13_857",
    ),
    "ic15": (
        "atmodel_data/ocr_datasets/evaluation/IC15_1811",
    ),
    "cute80": (
        "atmodel_data/ocr_datasets/evaluation/CUTE80",
    ),
}

class OCRDataset(Dataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 is_train=True,
                 ):
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.data_aug = cfg['INPUT']['DATA_AUG']
        self.img_h, self.img_w = cfg['INPUT']['IMAGE_SIZE'][0], cfg['INPUT']['IMAGE_SIZE'][1]
        self.convert_mode = cfg['INPUT']["CONVERT_MODE"]
        self.check_length = cfg['INPUT']["CHECK_LENGTH"]
        self.multiscales = cfg['INPUT']["MULTISCALES"]
        self.max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        self.tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])

        path = OCR_DATASETS[self.dataset_name][0]
        self.path, self.name = Path(path), Path(path).name

        charset_path = cfg['INPUT']['CHARSET_PATH']
        self.charset = CharsetMapper(charset_path, max_length=self.max_token_num)
        self.character = self.charset.label_to_char.values()

        # load data
        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {path}.'
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))

        if self.is_train and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return self.length

    def _next_image(self, index):
        next_index = random.randint(0, len(self) - 1)
        return self.get(next_index)

    def _check_image(self, x, pixels=6):
        if x.size[0] <= pixels or x.size[1] <= pixels:
            return False
        else:
            return True

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.img_w < self.img_h:
                if fix_h:
                    trg_h = self.img_h
                else:
                    trg_h = int(ratio * self.img_w)
                trg_w = self.img_w
            else:
                trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_w - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            return img

        if self.is_train:
            if random.random() < 0.5:
                base, maxh, maxw = self.img_h, self.img_h, self.img_w
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h / w)
            else:
                return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        else:
            return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    def resize(self, img):
        if self.multiscales:
            return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
        else:
            return cv2.resize(img, (self.img_w, self.img_h))

    def get(self, idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
            try:
                label = str(txn.get(label_key.encode()), 'utf-8').strip()  # label
                if not set(label).issubset(self.character):
                    return self._next_image(idx)
                # label = re.sub('[^0-9a-zA-Z]+', '', label)
                if self.check_length and self.max_token_num > 0:
                    if len(label) > self.max_token_num or len(label) <= 0:
                        # logging.info(f'Long or short text image is found: {self.name}, {idx}, {label}, {len(label)}')
                        return self._next_image(idx)
                label = label[:self.max_token_num]

                imgbuf = txn.get(image_key.encode())  # image
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                    image = Image.open(buf).convert(self.convert_mode)
                if self.is_train and not self._check_image(image):
                    # logging.info(f'Invalid image is found: {self.name}, {idx}, {label}, {len(label)}')
                    return self._next_image(idx)
            except:
                import traceback
                traceback.print_exc()
                logging.info(f'Corrupted image is found: {self.name}, {idx}, {label}, {len(label)}')
                return self._next_image(idx)
            return image, label, idx

    def _process_training(self, image):
        if self.data_aug: image = self.augment_tfs(image)
        image = self.resize(np.array(image))
        return image

    def _process_test(self, image):
        return self.resize(np.array(image))  # TODO:move is_train to here

    def __getitem__(self, idx):
        image, text, idx_new = self.get(idx)
        # print(image, text, idx_new, idx)
        if not self.is_train: assert idx == idx_new, f'idx {idx} != idx_new {idx_new} during testing.'

        if self.is_train:
            image = self._process_training(image)
        else:
            image = self._process_test(image)
        # if self.return_raw: return {"image": image, "text": text, "width": img_w, "height": img_h}
        dataset_dict ={}
        image = self.totensor(image)
        question = ["What is the text on this image?".lower()]
        answers = [' '.join(text.lower())]

        dataset_dict['width'] = image.shape[2] 
        dataset_dict['height'] = image.shape[1]
        dataset_dict['image'] = image
        dataset_dict['question'] = question
        dataset_dict['answers'] = answers
        question_token = self.tokenizer(
            question, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )
        answer_token = self.tokenizer(
            answers, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )


        dataset_dict['question_tokens'] = {"input_ids": question_token["input_ids"], "attention_mask": question_token["attention_mask"]}
        dataset_dict['answer_tokens'] = {"input_ids": answer_token["input_ids"], "attention_mask": answer_token["attention_mask"]}

        # return image, y
        return dataset_dict
