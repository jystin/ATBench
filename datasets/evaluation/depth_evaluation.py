import logging
import torch
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.evaluation.evaluator import DatasetEvaluator

from itertools import chain


class DepthEvaluator(DatasetEvaluator):
    """
    Evaluate depth map metrics.
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
        self.d1 = []
        self.d2 = []
        self.d3 = []
        self.abs_rel = []
        self.sq_rel = []
        self.rmse = []
        self.rmse_log = []
        self.log_10 = []
        self.silog = []

    def process(self, inputs, outputs):
        # evaluate depth map
        for input, output in zip(inputs, outputs):
            pred = output["depth_map"]
            target = input["depth_map"]
            max_depth_eval = input.get("depth_max_depth", 10.0)
            min_depth_eval = input.get("depth_min_depth", 1e-3)
            pred, target = self.cropping_img(pred, target, max_depth_eval=max_depth_eval, min_depth_eval=min_depth_eval)
            d1, d2,  d3, abs_rel, sq_rel, rmse, rmse_log, log_10, silog = self.eval_depth(pred, target)
            self.d1.append(d1)
            self.d2.append(d2)
            self.d3.append(d3)
            self.abs_rel.append(abs_rel)
            self.sq_rel.append(sq_rel)
            self.rmse.append(rmse)
            self.rmse_log.append(rmse_log)
            self.log_10.append(log_10)
            self.silog.append(silog)

    def cropping_img(self, pred, gt_depth, min_depth_eval, max_depth_eval):
        pred[torch.isinf(pred)] = max_depth_eval
        pred[torch.isnan(pred)] = min_depth_eval

        valid_mask = torch.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

        if self._dataset_name == "nyuv2_depth_val":
            eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
            eval_mask[45:471, 41:601] = 1
        elif self._dataset_name == "kitti_depth_val":
            raise NotImplementedError
        else:
            eval_mask = valid_mask

        valid_mask = torch.logical_and(valid_mask, eval_mask)

        return pred[valid_mask], gt_depth[valid_mask]

    def eval_depth(self, pred, target):
        assert pred.shape == target.shape, "Prediction and target must have the same shape, but got pred: {} and target: {}".format(pred.shape, target.shape)

        thresh = torch.max((target / pred), (pred / target))

        d1 = torch.sum(thresh < 1.25).float() / len(thresh)
        d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
        d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

        diff = pred - target
        diff_log = torch.log(pred) - torch.log(target)

        abs_rel = torch.mean(torch.abs(diff) / target)
        sq_rel = torch.mean(torch.pow(diff, 2) / target)

        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
        rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

        log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
        silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

        return (d1.cpu(), d2.cpu(), d3.cpu(), abs_rel.cpu(), sq_rel.cpu(), rmse.cpu(), rmse_log.cpu(), log10.cpu(), silog.cpu())

    def evaluate(self):
        if self._distributed:
            synchronize()
            def gather(x, move=False):
                x = all_gather(x)
                x = list(chain(*x))
                if move:
                    x = [xx.to(self.d1[0].device) for xx in x]
                return x
            self.d1 = gather(self.d1)
            self.d2 = gather(self.d2)
            self.d3 = gather(self.d3)
            self.abs_rel = gather(self.abs_rel)
            self.sq_rel = gather(self.sq_rel)
            self.rmse = gather(self.rmse)
            self.rmse_log = gather(self.rmse_log)
            self.log_10 = gather(self.log_10)
            self.silog = gather(self.silog)
            if not is_main_process():
                return {}

        d1 = (sum(self.d1) / len(self.d1)).item()
        d2 = (sum(self.d2) / len(self.d2)).item()
        d3 = (sum(self.d3) / len(self.d3)).item()
        abs_rel = (sum(self.abs_rel) / len(self.abs_rel)).item()
        sq_rel = (sum(self.sq_rel) / len(self.sq_rel)).item()
        rmse = (sum(self.rmse) / len(self.rmse)).item()
        rmse_log = (sum(self.rmse_log) / len(self.rmse_log)).item()
        log_10 = (sum(self.log_10) / len(self.log_10)).item()
        silog = (sum(self.silog) / len(self.silog)).item()

        result = {"d1": d1, "d2": d2, "d3": d3, "abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse, "rmse_log": rmse_log, "log_10": log_10, "silog": silog}
        return result

