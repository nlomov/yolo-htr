# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import OBBPredictor

        args = dict(model='yolov8n-obb.pt', source=ASSETS)
        predictor = OBBPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes OBBPredictor with optional model and data configuration overrides."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"

    def postprocess(self, preds, img, orig_imgs, true_preds=None):
        """Post-processes predictions and returns a list of Results objects."""
        feats = preds[1][0][-1]
        
        if true_preds is None:
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=len(self.model.names),
                classes=self.args.classes,
                rotated=True,
            )
        else:
            preds = [pred.clone() for pred in true_preds]
            for i in range(len(preds)):
                gain = min(img.shape[2] / orig_imgs[i].shape[0], img.shape[3] / orig_imgs[i].shape[1])  # gain  = old / new
                pad = (
                    round((img.shape[3] - orig_imgs[i].shape[1] * gain) / 2 - 0.1),
                    round((img.shape[2] - orig_imgs[i].shape[0] * gain) / 2 - 0.1),
                )
                preds[i][:,:4] *= gain
                preds[i][:,:4] += torch.tensor([pad[0],pad[1],pad[0],pad[1]])
                preds[i] = preds[i].to(feats.dtype).to(self.model.device)
            
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        i = 0
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            
            pred_cells = pred[:, :4] / 8
            pred_angles = pred[:, 6]
            
            if not true_preds:
                rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
                rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
                # xywh, r, conf, cls
                obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
                results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
            else:
                results.append(Results(orig_img, path=img_path, names=self.model.names, obb=true_preds[i][:,(0,1,2,3,6,4,5)]))
            
            char_probs = ops.scores_by_obb(pred_cells, pred_angles, feats[i])
            char_probs = self.model.model.model[-1](char_probs.to(self.model.model.model[-1].end_conv.weight.dtype), \
                                                    groups=self.model.model.ng if hasattr(self.model.model,'ng') else 0)
            
            lines = ops.decode_probs(char_probs, sorted(self.model.model.charset))            
            for i in range(len(lines)):
                lines[i] = ''.join(chr(ord(c) % 10000) + \
                               ('' if ord(c) < 10000 else (chr(0)+chr(818)+chr(821)+chr(819))[ord(c) // 10000]) for c in lines[i])
            results[-1].lines = lines
            i += 1
            
        return results
