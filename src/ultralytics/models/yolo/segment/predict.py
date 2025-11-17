# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
import torch
import itertools
from ultralytics.data.utils import polygons2masks
import copy
import numpy as np
import cv2
from matplotlib.path import Path


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

    def postprocess(self, preds, img, orig_imgs, true_preds=None):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        nc = len(self.model.names)
        
        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        
        if not true_preds:
            p = ops.non_max_suppression(
                preds[0],
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=nc,
                classes=self.args.classes,
            )
        else:
            temp = [pred.clone() for pred in true_preds[0]]
            for i in range(len(temp)):
                gain = min(img.shape[2] / orig_imgs[i].shape[0], img.shape[3] / orig_imgs[i].shape[1])  # gain  = old / new
                pad = (
                    round((img.shape[3] - orig_imgs[i].shape[1] * gain) / 2 - 0.1),
                    round((img.shape[2] - orig_imgs[i].shape[0] * gain) / 2 - 0.1),
                )
                temp[i][:,:4] *= gain
                temp[i][:,:4] += torch.tensor([pad[0],pad[1],pad[0],pad[1]])
                temp[i] = temp[i].to(preds[-1][0][-1].dtype).to(self.model.device)
            p = temp
            
            true_masks = copy.deepcopy(true_preds[1])
            for i in range(len(true_masks)):
                for j in range(len(true_masks[i])):
                    true_masks[i][j] *= gain
                    true_masks[i][j] += pad
            
            X,Y = np.meshgrid(np.arange(2,img.shape[3],4),np.arange(2,img.shape[2],4))
            
            temp = [np.zeros((len(pols),X.shape[0],X.shape[1]),dtype='bool') for pols in true_masks]
            for i,pols in enumerate(true_masks):
                for j,pol in enumerate(pols):
                    xidx = np.where((X[0,:] >= pol[:,0].min()) & (X[0,:] <= pol[:,0].max()))[0]
                    yidx = np.where((Y[:,0] >= pol[:,1].min()) & (Y[:,0] <= pol[:,1].max()))[0]
                    if xidx.size > 0 and yidx.size > 0:
                        P = [(x,y) for x,y in zip(X[np.ix_(yidx,xidx)].flatten(), Y[np.ix_(yidx,xidx)].flatten())]
                        temp[i][np.ix_([j],yidx,xidx)] = Path(pol).contains_points(P).reshape((len(yidx),len(xidx)))
            true_masks = [torch.tensor(mask).to(proto.dtype).to(proto.device) for mask in temp]
        
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        
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
                if type(self.model.model.model[-3]).__name__ == 'SegmentOld':
                    if not true_preds:
                        masks = ops.process_mask_old(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=False)  # HWC
                    else:
                        masks = true_masks[i]
                    char_probs = ops.scores_by_mask_old(pred_cells, p[i][:,6:], proto[i], preds[1][0][-1][i])
                else:
                    if not true_preds:
                        masks = ops.process_mask(proto[i], self.model.model.stride[-1], pred[:, :4], img.shape[2:], upsample=False)
                    else:
                        masks = true_masks[i]
                    char_probs = ops.scores_by_mask(pred_cells, masks, preds[1][0][-1][i])
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            
            if not true_preds:
                old_shape = masks.shape[1:]
                new_shape = orig_img.shape[:2]
                temp = np.zeros((masks.shape[0],masks.shape[1]+1,masks.shape[2]+1))
                temp[:,:-1,:-1] = masks.detach().cpu().numpy()
                temp = np.stack([cv2.dilate(temp[i],np.array([[1,1],[1,1]])) for i in range(temp.shape[0])])
                temp = [ops.scale_coords(old_shape, x, new_shape, normalize=False) for x in ops.masks2segments(torch.tensor(temp))]
                temp = {'xy': temp}
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
            else:
                temp = {'xy': true_preds[1][i]}
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=true_preds[0][i]))
            temp = type('MyMasks', (object,), temp)
            results[-1].masks = temp
            if len(pred):
                dtype = self.model.model.model[-1].end_conv.weight.dtype
                char_probs = self.model.model.model[-1](char_probs.to(dtype), groups=self.model.model.ng)

                lines = ops.decode_probs(char_probs, sorted(self.model.model.charset))
                results[-1].lines = lines
            else:
                results[-1].lines = []
            
        return results
