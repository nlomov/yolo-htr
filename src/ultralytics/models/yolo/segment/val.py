# Ultralytics YOLO 🚀, AGPL-3.0 license

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou, get_cer_and_wer
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.ops import scores_by_mask, decode_probs, format_string_for_wer
import itertools
from editdistance import distance as editdistance
from scipy.optimize import linear_sum_assignment


class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml')
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """Initialize metrics and select mask processing function based on save_json flag."""
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[])

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 12) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "CER",
            "WER",
        )

    def postprocess(self, preds, true_bboxes=None):
        """Post-processes YOLO predictions and returns output detections with proto."""
        p0 = ops.non_max_suppression(
             preds[0],
             self.args.conf,
             self.args.iou,
             labels=self.lb,
             multi_label=True,
             agnostic=self.args.single_cls,
             max_det=self.args.max_det,
             nc=self.nc,
        )
        
        proto0 = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        
        if true_bboxes is not None:
            boxes, idx, proto = true_bboxes
            p = []
            for i in range(len(p0)):
                p.append(boxes[idx == i])
        else:
            p, proto = p0, proto0
                
        try:
            chars = sorted(self.model.model.charset)
        except:
            chars = sorted(self.model.charset)
        pred_enc = []
        
        for i in range(len(p)):
            char_probs = scores_by_mask(p[i][:,:4] / 8, p[i][:,6:], proto[i], preds[1][0][-1][i])
            try:
                dtype = self.model.model.model[-1].end_conv.weight.dtype
                char_probs = self.model.model.model[-1](char_probs.to(dtype))
            except:
                dtype = self.model.model[-1].end_conv.weight.dtype
                char_probs = self.model.model[-1](char_probs.to(dtype))
            lines = decode_probs(char_probs, chars)
            pred_enc.append(lines)
            
        return (p0, proto0), pred_enc, char_probs

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by processing images and targets."""
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        """Prepares a batch for training or inference by processing images and targets."""
        predn = super()._prepare_pred(pred, pbatch)
        pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch["imgsz"])
        return predn, pred_masks

    def update_metrics(self, preds, pred_enc, batch):
        """Metrics."""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Masks
            gt_masks = pbatch.pop("masks")
            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_m"] = self._process_batch(
                    predn, bbox, cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True
                )
                
                if self.args.ctc > 0:
                    true_lines = [batch["lines"][i] for i,idx in enumerate(batch['batch_idx']) if idx == si]
                    if not self.args.save_preds:
                        conf_thr = 0.5
                        iou_thr = 0.5
                        pred_lines = [pred_enc[si][i] for i,val in enumerate(predn[:,4]) if val >= conf_thr]
                        #cost_matrix = np.array([editdistance(a,b) for a,b in itertools.product(true_lines, pred_lines)])
                        cost_matrix = box_iou(bbox, predn[predn[:,4] >= conf_thr, :4]).detach().cpu().numpy()
                        if cost_matrix.size > 0:
                            cost_matrix = cost_matrix * (cost_matrix >= iou_thr)
                            true_idx, pred_idx = linear_sum_assignment(cost_matrix, maximize=True)
                            filt = [(t,p) for t,p in zip(true_idx,pred_idx) if cost_matrix[t,p] >= iou_thr]
                            if len(filt) > 0:
                                true_idx, pred_idx = list(zip(*filt))
                                true_idx, pred_idx = list(true_idx), list(pred_idx)
                            else:
                                true_idx, pred_idx = [], []
                        else:
                            true_idx, pred_idx = [], []
                    else:
                        pred_lines = pred_enc[si]
                        true_idx, pred_idx = np.arange(len(true_lines)), np.arange(len(true_lines))
                            
                    cer, wer, clen, wlen = 0, 0, 0, 0
                    
                    for i,j in zip(true_idx, pred_idx):
                        pred_chars, true_chars = pred_lines[j], true_lines[i]
                        pred_words = [w for w in format_string_for_wer(pred_chars).split(' ') if len(w) > 0]
                        true_words = [w for w in format_string_for_wer(true_chars).split(' ') if len(w) > 0]
                        clen += len(true_chars)
                        wlen += len(true_words)
                        if self.data['wc'] in true_chars:    
                            #cer += editdistance_chars(pred_chars, true_chars)
                            #wer += editdistance_words(pred_words, true_words)
                            pass
                        else:
                            cer += editdistance(pred_chars, true_chars)
                            wer += editdistance(pred_words, true_words)
                    '''
                    for i in range(cost_matrix.shape[0]):
                        if i not in labels_idx:
                            true_chars = true_lines[i]
                            true_words = [w for w in format_string_for_wer(true_chars).split(' ') if len(w) > 0]
                            cer += len(true_chars)
                            wer += len(true_words)
                            clen += len(true_chars)
                            wlen += len(true_words)
                    '''
                    
                    self.cer.append(cer/max(clen,1))
                    self.wer.append(wer/max(wlen,1))
                else:
                    self.cer.append(1.0)
                    self.wer.append(1.0)
                
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:].cpu())  # filter top 15 to plot

            # Save
            if self.args.save_json:
                pred_masks = ops.scale_image(
                    pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                    pbatch["ori_shape"],
                    ratio_pad=batch["ratio_pad"][si],
                )
                self.pred_to_json(predn, batch["im_file"][si], pred_masks)
            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def finalize_metrics(self, *args, **kwargs):
        """Sets speed and confusion matrix for evaluation metrics."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Return correct prediction matrix.

        Args:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2

        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks and bounding boxes."""
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=self.args.max_det),  # not set to self.args.max_det due to slow plotting speed
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
        self.plot_masks.clear()

    def pred_to_json(self, predn, filename, pred_masks):
        """
        Save one JSON result.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )

    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                        :2
                    ]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
