import os
import json
import logging
import itertools

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator

from caption_pycocotools.vqa import VQA
from pycocoevalcap.vqaEval import VQAEval
from vizwiz_api.vqa import Vizwiz_VQA
from vizwiz_eval_cap.vqaEval import Vizwiz_VQAEval


class VQAEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).
    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        anno_json=None,
        question_json=None
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self.anno_json = anno_json
        self.question_json = question_json
        if "vqav2" in self._dataset_name:
            self._gt_json = VQA(anno_json, question_json)
        else:
            self._gt_json = Vizwiz_VQA(anno_json)

    def reset(self):
        self._gen_vqas = []
        self._images = []
        self._gen_answerable = []
        self._question_ids = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for output in outputs:
            self._images.append(output['image_name'])
            self._gen_vqas.append(output['vqa_text'])
            self._gen_answerable.append(output['answerable'])
            self._question_ids.append(output['question_id'])

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        if self._distributed:
            comm.synchronize()
            def gather(x, move=False):
                x = comm.gather(x)
                x = list(itertools.chain(*x))
                if move:
                    x = [xx.to(self._gen_vqas[0].device) for xx in x]
                return x
            gen_vqas = gather(self._gen_vqas)
            images = gather(self._images)
            answerable = gather(self._gen_answerable)
            question_ids = gather(self._question_ids)
            if not comm.is_main_process():
                return {}
        else:
            gen_vqas = self._gen_vqas
            images = self._images
            answerable = self._gen_answerable
            question_ids = self._question_ids


        assert len(gen_vqas) == len(images)
        gt_answers = self._gt_json

        # Evaluate the results
        if "vqav2" in self._dataset_name:
            pred_vqa = [{"question_id": question_ids[i], "answer": gen_vqas[i]} for i in range(len(gen_vqas))]
            pred_vqa_pth = os.path.join(self._output_dir, 'pred_vqav2_results.json')
            json.dump(pred_vqa, open(pred_vqa_pth, "w"))

            vqaRes = VQA(pred_vqa_pth, self.question_json)
            vqaEval = VQAEval(gt_answers, vqaRes, n=2)
            vqaEval.evaluate()
            result = dict(
                accuracy=vqaEval.accuracy,
                evalQA=vqaEval.evalQA,
                evalAnsType=vqaEval.evalAnsType,
                evalQuesType=vqaEval.evalQuesType,
            )

        else:
            pred_vqa = [{"image": image, "answer": gen_vqas[i], "answerable": answerable[i]} for i, image in
                        enumerate(images)]
            pred_vqa_pth = os.path.join(self._output_dir, 'pred_vizwiz_vqa_results.json')
            json.dump(pred_vqa, open(pred_vqa_pth, "w"))

            vqaRes = Vizwiz_VQA(pred_vqa_pth)
            vqaEval = Vizwiz_VQAEval(gt_answers, vqaRes, n=2)
            vqaEval.evaluate()
            vqaEval.evaluate_unanswerability()
            result = dict(
                accuracy=vqaEval.accuracy,
                caption_metric=vqaEval.caption_metric,
                evalQA=vqaEval.evalQA,
                evalAnsType=vqaEval.evalAnsType,
                unanswerability=vqaEval.unanswerability,
            )
        return result