import logging
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.evaluation.evaluator import DatasetEvaluator

from itertools import chain


class OcrEvaluator(DatasetEvaluator):
    """
    Evaluate ocr metrics.
    """
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

    def reset(self):
        self.acc = []   

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            pred = output["ocr_text"]
            gt = input["answers"][0]
            self.acc.append(1.0 if pred == gt else 0.0)

    def evaluate(self):
        if self._distributed:
            synchronize()
            self.acc = all_gather(self.acc)
            self.acc = list(chain(*self.acc))
            if not is_main_process():
                return {}

        acc = sum(self.acc) / len(self.acc)

        return {"ocr_acc": acc, "len": len(self.acc)}