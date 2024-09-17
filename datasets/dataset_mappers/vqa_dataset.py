from detectron2.utils.file_io import PathManager
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from atmodel.language.LangEncoder import build_tokenizer

from fastai.vision import *
import string


VQA_DATASETS = {
    "vizwiz_vqa_train": (
        # image root
        "vqa_datasets/vizwiz/train",
        # annotation
        "vqa_datasets/vizwiz/annotations/train.json",
    ),
    "vizwiz_vqa_val": (
        "vqa_datasets/vizwiz/val",
        "vqa_datasets/vizwiz/annotations/val.json",
    ),
    "vizwiz_vqa_test": (
        "vqa_datasets/vizwiz/test",
        "vqa_datasets/vizwiz/annotations/test.json",

    ),
}


class VQADataset(Dataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 is_train=True):
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.image_format = cfg['INPUT']['FORMAT']
        self.max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        self.tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        self.pre_vocabs_path = cfg['INPUT']['PATH_VOCABS']

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
        image_root, annotation_file = VQA_DATASETS[dataset_name]

        with PathManager.open(os.path.join(root, annotation_file), "r") as f:
            annos = json.load(f)

        pre_vocab_path = os.path.join(root, self.pre_vocabs_path )
        pre_vocab = self.get_vocab(pre_vocab_path)

        split = dataset_name.split("_")[-1]


        dataset_info = {
                "split": split,
                "pre_vocabs": pre_vocab,
                "image_root": os.path.join(root, image_root),
                "annotation_file": os.path.join(root, annotation_file),
                "annos": annos}


        return dataset_info

    def set_metadata(self, dataset_name, evaluator_type):
        MetadataCatalog.get(dataset_name).set(
            image_root=self.info['image_root'],
            gt_json=self.info['annotation_file'],
            question_json=None,
            evaluator_type=evaluator_type,
            **self.meta,
        )

    def __len__(self):
        return len(self.info["annos"])

    def __getitem__(self, idx):
        anno = self.info["annos"][idx]
        image_name = anno["image"]
        file_name = os.path.join(self.info["image_root"], image_name)
        question = pre_question_vizwiz(anno["question"])

        dataset_dict = {
            "image_name": image_name,
            "file_name": file_name,
            "question_id": -1,
            "question": [question],
        }

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        question_token = self.tokenizer(
            question, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )
        dataset_dict['question_tokens'] = {"input_ids": question_token["input_ids"],
                                           "attention_mask": question_token["attention_mask"]}

        if not self.is_train:
            return dataset_dict

        answer = find_most_common_answer(anno["answers"])
        answer = pre_answers_vizwiz(answer)
        if self.info["pre_vocabs"].get(answer) is None:
            answer = "unanswerable"
        answerable = anno["answerable"]
        answer_type = anno["answer_type"]
        dataset_dict["answers"] = [answer],
        dataset_dict["answerable"] = answerable
        dataset_dict["answer_type"] = answer_type

        answer_token = self.tokenizer(
            answer, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )

        dataset_dict["answerable"] = torch.as_tensor(dataset_dict["answerable"]).float()
        dataset_dict['answer_tokens'] = {"input_ids": answer_token["input_ids"],
                                         "attention_mask": answer_token["attention_mask"]}

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


def pre_question_vizwiz(question):
    question = question.lower()

    # define desired replacements here
    punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}
    # punctuation_dict = {'.': '', '_': ' ', '?': '', '-': ' ', '/': ' ', ',': ' '}
    conversational_dict = {"thank you": '', "thanks": '', "thank": '', "please": '', "hello": '',
                           "hi ": ' ', "hey ": ' ', "good morning": '', "good afternoon": '', "have a nice day": '',
                           "okay": '', "goodbye": ''}

    rep = punctuation_dict
    rep.update(conversational_dict)

    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    question = pattern.sub(lambda m: rep[re.escape(m.group(0))], question)
    question = question.lstrip(' ').rstrip(' ')
    question = question.strip(' ') + '?'

    return question

def pre_answers_vizwiz(answer):
    answer = answer.lower()

    # define desired replacements here
    punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}

    rep = punctuation_dict
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    answer = pattern.sub(lambda m: rep[re.escape(m.group(0))], answer)
    return answer

def find_most_common_answer(answers):
    answers = [answer["answer"] for answer in answers]
    answer_counter = Counter(answers)
    most_common_answers = answer_counter.most_common()
    most_common_answer, count = most_common_answers[0]
    return most_common_answer

