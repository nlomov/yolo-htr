# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import torch
import numpy as np

class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    
    def postprocess(self, preds, img, orig_imgs, true_preds=None):
        """Post-processes predictions and returns a list of Results objects."""
        feats = preds[1][-1]
        
        if not true_preds:
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
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
        for i, pred in enumerate(preds):
            pred_cells = pred[:, :4] / 8
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            
            if not true_preds:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            else:
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=true_preds[i]))
            
            char_probs = ops.scores_by_box(pred_cells, feats[i])
            char_probs = self.model.model.model[-1](char_probs)
            #results[-1].char_probs = char_probs.detach().cpu().numpy()
            lines = ops.decode_probs(char_probs, sorted(self.model.model.charset))
            results[-1].lines = lines
                
        return results
