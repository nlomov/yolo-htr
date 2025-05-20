# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
import torch
import itertools


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )
        
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            
        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        
        p = []
        for i in range(preds[0].shape[0]):
            temp = preds[0][i,:,preds[0][i,4] > self.args.conf].transpose(1,0)
            idx = temp[:,4].sort(descending=True)[1]
            temp = temp[idx]
            boxes = temp[:,:4] / 4
            masks = torch.einsum('in,nhw->ihw', temp[:,5:], proto[i]).sigmoid()
            X = torch.arange(proto.shape[-1]).to(self.device)
            Y = torch.arange(proto.shape[-2]).to(self.device)
            dl = (boxes[:, 0:1] - X).clip(0,1)
            dr = (X+1 - boxes[:, 2:3]).clip(0,1)
            dt = (boxes[:, 1:2] - Y).clip(0,1)
            db = (Y+1 - boxes[:, 3:4]).clip(0,1)
            hmask = (1 - dl - dr)
            vmask = (1 - dt - db)
            masks = (masks * vmask[...,None]) * hmask.unsqueeze(1)
            masks = masks[:,vmask.any(axis=0)][:,:,hmask.any(axis=0)]
            
            idx = []
            for i in range(masks.shape[0]):
                add = True
                for j in range(i):
                    iou = torch.minimum(masks[i],masks[j]).sum() / torch.maximum(masks[i],masks[j]).sum()
                    if iou > self.args.iou:
                        add = False
                        break
                if add:
                    idx.append(i)
            p.append(torch.hstack([temp[idx,:5], torch.zeros((len(idx),1)).to(self.device), temp[idx,5:]]))
        
        for i, pred in enumerate(p):
            pred_cells = p[i][:,:4] / 8
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
            
            char_probs = ops.scores_by_mask(pred_cells, p[i][:,6:], proto[i], preds[1][0][-1][i])
            dtype = self.model.model.model[-1].end_conv.weight.dtype
            char_probs = self.model.model.model[-1](char_probs.to(dtype))
                
            lines = ops.decode_probs(char_probs, sorted(self.model.model.charset))
            results[-1].lines = lines
            
        return results
