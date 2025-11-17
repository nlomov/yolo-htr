# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh, scores_by_box, scores_by_obb, scores_by_mask, decode_probs
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from .metrics import bbox_iou, probiou
from .tal import bbox2dist
from ultralytics.nn.modules.head import Detect
from editdistance import distance as editdistance
from scipy.optimize import linear_sum_assignment
import itertools
from ultralytics.utils.ops import true_segmentation


def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        probs = log_probs[:input_length, i].exp()
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = (targets_prime[:-2] != targets_prime[2:])
        alpha[0] *= 0
        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += alpha[:-1]
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
            alpha[0] *= 0
        losses.append(-alpha[-2:-1].sum().log()[None])
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        output = (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        output = output.sum()
    output = output.to(dt)
    return output


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        
        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        
        val = (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
        
        fake_dist = torch.zeros_like(pred_dist)
        fake_dist[np.arange(pred_dist.shape[0]), tl.view(-1)] = wl.view(-1).to(fake_dist.dtype)
        fake_dist[np.arange(pred_dist.shape[0]), tr.view(-1)] = wr.view(-1).to(fake_dist.dtype)
        fake_dist = torch.log(fake_dist)
        fake_dist[fake_dist.isinf()] = -10000
        
        ref = (
            F.cross_entropy(fake_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(fake_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
        
        return val-ref


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        
        # DFL loss
        if self.use_dfl:
            target_bboxes = target_bboxes.view(-1, 5)
            target_bboxes = target_bboxes[:,:4].view(pred_bboxes.shape[0], -1, 4)
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        idx = [i for i,m in enumerate(model.model) if isinstance(m, Detect)]
        
        m = model.model[idx[-1]]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.charset = model.charset
        self.last = model.model[-1]

        self.use_dfl = m.reg_max > 1
        self.true_boxes = h.true_boxes
        
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def emulate_assigner(self, stride_tensor, lines_enc, true_lines, batch_idx, gt_labels, gt_bboxes, \
                         pred_bboxes, pred_scores, pred_masks, probs, proto=None):
        with torch.no_grad():
            
            target_gt_idx = torch.zeros((pred_bboxes.shape[:2]), dtype=torch.int64, device=self.device)
            fg_mask = torch.zeros((pred_bboxes.shape[:2]), dtype=torch.bool, device=self.device)
            mask_pos = torch.zeros((pred_bboxes.shape[0], lines_enc.shape[1], pred_bboxes.shape[1]), dtype=torch.bool, device=self.device)
            align_metric = torch.zeros(mask_pos.shape, device=self.device)
            overlaps = torch.zeros(mask_pos.shape, device=self.device)
            pairs = []

            pred_cells = pred_bboxes[:,:,:4] * (stride_tensor / self.stride[0])
            for i in range(pred_scores.shape[0]):
                temp = torch.argsort(pred_scores[i,:,0], descending=True).detach().cpu().numpy()
                batch_lines = [line for line,idx in zip(true_lines,batch_idx) if idx == i]
                span = 2 * np.array([len(x)*self.stride[0] for x in batch_lines])

                mask_small = span <= (self.reg_max-1)*2*int(self.stride[0])
                mask_medium = span <= (self.reg_max-1)*2*int(self.stride[1])
                num_small = sum(mask_small) * self.assigner.topk
                num_medium = sum(mask_medium) * self.assigner.topk
                num_cands = 2 * len(batch_lines) * self.assigner.topk

                idx = []
                for t in temp:
                    if stride_tensor[t] == self.stride[0]:
                        if num_small > 0 and num_medium > 0:
                            num_small -= 1
                            num_medium -= 1
                            idx.append(t)
                    elif stride_tensor[t] == self.stride[1]:
                        if num_medium > 0:
                            num_medium -= 0
                            idx.append(t)
                    else:
                        idx.append(t)
                    if len(idx) == num_cands:
                        break
                idx = np.array(idx)

                lines = []
                chunk = 1000
                idx = idx[:num_cands]

                for j in range(int(np.ceil(num_cands / chunk))):
                    if proto:
                        char_probs = scores_by_mask(pred_cells[i,idx[j*chunk:(j+1)*chunk]], pred_masks[i,idx[j*chunk:(j+1)*chunk]], \
                                                    proto[i], probs[i])
                    else:
                        char_probs = scores_by_obb(pred_cells[i,idx[j*chunk:(j+1)*chunk]], pred_masks[i,idx[j*chunk:(j+1)*chunk]], probs[i])
                    char_probs = char_probs.to(self.last.end_conv.weight.dtype)
                    char_probs = self.last(char_probs)
                    lines += decode_probs(char_probs, sorted(self.charset))

                C = [0 if b == '@' else editdistance(a,b) for a,b in itertools.product(lines, batch_lines)]
                C = np.array(C).reshape(len(lines), len(batch_lines))
                C = np.maximum(1 - C / [len(x) for x in batch_lines], 0.0)

                overlaps[i,:C.shape[1],idx] = torch.tensor(C.transpose(), dtype=torch.float32, device=self.device)
                C = pred_scores[i,idx].sigmoid().pow(self.assigner.alpha).cpu().numpy() * np.power(C, self.assigner.beta)
                align_metric[i,:C.shape[1],idx] = torch.tensor(C.transpose(), dtype=torch.float32, device=self.device)

                C[(stride_tensor[idx] == self.stride[0]).detach().cpu().numpy() * np.logical_not(mask_small)] = -100000
                C[(stride_tensor[idx] == self.stride[1]).detach().cpu().numpy() * np.logical_not(mask_medium)] = -100000
                C = np.tile(C, [1,self.assigner.topk])
                [row_ind,col_ind] = linear_sum_assignment(C, maximize=True)
                pairs.append(idx[np.array([(a[0],b[0]) for a,b in itertools.combinations(zip(row_ind,col_ind), 2) \
                                          if a[1] % len(batch_lines) == b[1] % len(batch_lines)])])

                target_gt_idx[i,idx[row_ind]] = torch.tensor(col_ind % len(batch_lines), device=self.device)
                fg_mask[i,idx[row_ind]] = True
                mask_pos[i,torch.tensor(col_ind % len(batch_lines)),idx[row_ind]] = True

            self.assigner.bs = mask_pos.shape[0]
            self.assigner.n_max_boxes = mask_pos.shape[1]

            # Assigned target
            _, _, target_scores, lines_enc = self.assigner.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask, lines_enc)

            # Normalize
            align_metric *= mask_pos
            pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
            pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.assigner.eps)).amax(-2).unsqueeze(-1)
            target_scores = target_scores * norm_align_metric
            
        return target_scores, lines_enc, fg_mask, pairs
        
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""        
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, ctc
        
        if isinstance(preds, tuple):
            preds = preds[1]
        feats = preds[:-1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        lines_enc0 = torch.zeros((targets.shape[0], targets.shape[1], batch["lines_enc"].shape[1]), dtype=torch.long)
        for i in range(batch_size):
            matches = batch["batch_idx"] == i
            n = matches.sum()
            if n:
                lines_enc0[i,:n] = batch["lines_enc"][matches.cpu()]
        
        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)        
        
        mask_size = ((gt_bboxes[:,:,2] - gt_bboxes[:,:,0]).unsqueeze(-1) <= (stride_tensor[:,0] * 2 * (self.reg_max - 1)))
        mask_size[:,:,stride_tensor[:,0] == stride_tensor.max()] = True
        
        _, target_bboxes, target_scores, fg_mask, _, lines_enc = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            lines_enc0.to(self.device),
            mask_size,
            )
        target_scores_sum = max(target_scores.sum(), 1)
        
        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE        

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes/stride_tensor, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        
        if self.true_boxes:
            pred_bboxes = gt_bboxes.to(self.device)
            target_scores = mask_gt.to(self.device)
            fg_mask = mask_gt[...,0].to(self.device) > 0
            lines_enc = lines_enc0.to(self.device)
        
        if self.hyp.ctc > 0 and fg_mask.sum():
            val = torch.tensor(0.0).to(self.device)
            target_weights_sum = torch.tensor(0.0).to(self.device)
            for i in range(target_scores.shape[0]):
                index = fg_mask[i]
                pred_cells = pred_bboxes[i,index] / stride_tensor[0]
                
                char_probs = scores_by_box(pred_cells, preds[-1][i])
                char_probs = self.last(char_probs)
                char_probs = torch.nn.functional.log_softmax(char_probs.permute(1,0,2), dim=-1)
                
                ctc_loss = torch.nn.CTCLoss(blank=char_probs.shape[-1]-1, reduction="none")
                input_lengths = char_probs.shape[0] * torch.ones(char_probs.shape[1], dtype=torch.long).to(char_probs.device)
                target_enc = lines_enc[i,index]
                target_lengths = (target_enc >= 0).sum(axis=-1)
                
                torch.cuda.empty_cache()
                enc_sh = self.charset['#'] if '#' in self.charset else -2
                enc_at = -3
                
                wc_mask = (target_enc == enc_sh).any(dim = 1)
                if False:
                    target_enc[target_enc == enc_sh] = -1
                    target_enc[target_enc == enc_at] = -2
                    
                    temp = ctcloss_reference(char_probs[:,wc_mask].float(), target_enc[wc_mask], input_lengths[wc_mask], \
                                             target_lengths[wc_mask], blank=char_probs.shape[-1]-1, reduction="none")
                    target_weights = target_scores[i,index,0][wc_mask]
                    val += (temp * target_weights).sum()
                    target_weights_sum += target_weights.sum()
                
                wc_mask = torch.logical_not(torch.logical_or((target_enc == enc_sh).any(dim = 1), (target_enc == enc_at).any(dim = 1)))
                if wc_mask.any():
                    
                    temp = ctc_loss(char_probs[:,wc_mask].float(), target_enc[wc_mask], input_lengths[wc_mask], target_lengths[wc_mask])
                    target_weights = target_scores[i,index,0][wc_mask]
                    val += (temp * target_weights).sum()
                    target_weights_sum += target_weights.sum()
            
            loss[3] = self.hyp.ctc * val / max(target_weights_sum, 1)
        
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    
class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask
        
    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        
        probs = feats[-1]
        feats = feats[:-1]
        
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
            
        lines_enc0 = torch.zeros((targets.shape[0], targets.shape[1], batch["lines_enc"].shape[1]), dtype=torch.long)
        for i in range(batch_size):
            matches = batch["batch_idx"] == i
            n = matches.sum()
            if n:
                lines_enc0[i,:n] = batch["lines_enc"][matches.cpu()]
        lines_enc0 = lines_enc0.to(self.device)
        
        boxless = batch['im_file'][0].split('/')[-1][0] == '4'
        boxless = False
        
        if not boxless:
            mask_size = ((gt_bboxes[:,:,2] - gt_bboxes[:,:,0]).unsqueeze(-1) <= (stride_tensor[:,0] * 2 * (self.reg_max - 1)))
            mask_size[:,:,stride_tensor[:,0] == stride_tensor.max()] = True
            _, target_bboxes, target_scores, fg_mask, target_gt_idx, lines_enc = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
                lines_enc0,
                mask_size,
            )
            target_scores_sum = max(target_scores.sum(), 1)
            
            # Cls loss
            # loss[2] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
            loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
            
            if fg_mask.sum():
                # Bbox loss
                loss[0], loss[3] = self.bbox_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes / stride_tensor,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                )

                if self.hyp.seg:
                    val = 0
                    target_weights_sum = 0
                    for i in range(batch_size):
                        index = fg_mask[i]
                        gt_idx = target_gt_idx[i,index]
                        boxes = pred_bboxes[i,index,:4]
                        boxes *= stride_tensor[index] / 4
                        masks = torch.einsum('in,nhw->ihw', pred_masks[i,index], proto[i]).sigmoid()

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

                        pairs = [(i,j) for i,j in itertools.combinations(range(len(gt_idx)), r=2) if i != j and gt_idx[i] == gt_idx[j]]
                        idx1,idx2 = zip(*pairs)
                        idx1,idx2 = list(idx1),list(idx2)

                        scores = (target_scores[i,index,0][idx1] * target_scores[i,index,0][idx2])
                        masks1 = masks[idx1]
                        masks2 = masks[idx2]
                        iou = 1 - torch.minimum(masks1, masks2).sum(axis=(1,2)) / torch.maximum(masks1, masks2).sum(axis=(1,2))
                        val += (iou * scores).sum()
                        target_weights_sum += scores.sum()

                    loss[1] = val / max(target_weights_sum, 1)    
                
                    '''
                    # Masks loss
                    masks = batch["masks"].to(self.device).float()
                    if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                        masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

                    loss[1] = self.calculate_segmentation_loss(
                        fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
                    )
                    '''
            # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
                
            if self.true_boxes:
                pred_bboxes = target_bboxes.to(self.device)
                '''
                h,w = batch['resized_shape'][0]
                true_bboxes = batch['bboxes'].to(self.device) * torch.tensor([w,h,w,h]).to(self.device)
                true_bboxes = xywh2xyxy(true_bboxes)
                true_masks, proto = true_segmentation(true_bboxes, batch['masks'], batch['batch_idx'], proto.shape[1])
                temp_idx = target_gt_idx             
                for i in range(1, temp_idx.shape[0]):
                    temp_idx[i] += (batch['batch_idx'] < i).sum() 
                pred_bboxes = target_bboxes[:,:,:4].to(pred_bboxes.dtype)
                pred_masks = true_masks[temp_idx]
                '''
          
        if boxless:
            pred_bboxes /= stride_tensor
            target_scores,lines_enc,fg_mask,pairs = self.emulate_assigner(stride_tensor,lines_enc0,batch['lines'],batch['batch_idx'],\
                                                                          gt_labels,gt_bboxes,pred_bboxes,pred_scores,pred_masks,probs,proto)
            pred_bboxes *= stride_tensor
            
        if self.hyp.ctc > 0:
            val = torch.tensor(0.0).to(self.device)
            target_weights_sum = torch.tensor(0.0).to(self.device)
            for i in range(batch_size):
                index = fg_mask[i]
                pred_cells = pred_bboxes[i,index] / self.stride[0]

                char_probs = scores_by_mask(pred_cells, pred_masks[i,index], proto[i], probs[i])
                char_probs = self.last(char_probs)
                char_probs = torch.nn.functional.log_softmax(char_probs.permute(1,0,2), dim=-1)

                ctc_loss = torch.nn.CTCLoss(blank=char_probs.shape[-1]-1, reduction="none")
                input_lengths = char_probs.shape[0] * torch.ones(char_probs.shape[1], dtype=torch.long).to(char_probs.device)
                target_enc = lines_enc[i,index]

                target_lengths = (target_enc >= 0).sum(axis=-1)
                torch.cuda.empty_cache()
                enc_sh = self.charset['#'] if '#' in self.charset else -3
                enc_at = self.charset['@'] if '@' in self.charset else -2

                wc_mask = torch.logical_or((target_enc == enc_sh).any(dim = 1), (target_enc == enc_at).any(dim = 1))
                wc_mask = torch.logical_not(wc_mask)
                if wc_mask.any():
                    temp = ctc_loss(char_probs[:,wc_mask].float(), target_enc[wc_mask], input_lengths[wc_mask], target_lengths[wc_mask])
                    target_weights = target_scores[i,index,0][wc_mask]
                    val += (temp * target_weights).sum()
                    target_weights_sum += target_weights.sum()
            loss[4] = val / max(target_weights_sum, 1)

        if boxless:
            if self.hyp.box > 0:
                val = 0
                target_weights_sum = 0
                for i in range(batch_size):
                    idx1,idx2 = pairs[i][:,0],pairs[i][:,1]
                    boxes1,boxes2 = pred_bboxes[i,idx1,:4],pred_bboxes[i,idx2,:4]
                    boxes1 *= stride_tensor[idx1]
                    boxes2 *= stride_tensor[idx2]
                    temp = bbox_iou(boxes1,boxes2)
                    scores = target_scores[i,idx1] * target_scores[i,idx2]
                    val += ((1 - temp) * scores).sum()
                    target_weights_sum += scores.sum()   
                loss[0] = val / max(target_weights_sum, 1)
        
            if self.hyp.cls > 0:
                loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / max(target_scores.sum(), 1)    
            
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.seg  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain
        loss[4] *= self.hyp.ctc  # ctc loss

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_angles = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angles.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats[:-1]], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        
        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angles = pred_angles.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            tiny_mask = ((rw >= 2) & (rh >= 2)).cpu()
            targets = targets[tiny_mask]  # filter rboxes of tiny size to stabilize training
            batch_idx = batch_idx[tiny_mask, 0]
            batch_enc = batch["lines_enc"][tiny_mask]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
            
            gt_bboxes = gt_bboxes.view((-1,5))
            mask = gt_bboxes[:, 4] > torch.pi/4
            gt_bboxes[mask, 2:4] = gt_bboxes[mask][:, (3,2)]
            gt_bboxes[mask, 4] -= torch.pi/2
            gt_bboxes = gt_bboxes.view((gt_labels.shape[0],-1,5))
            
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angles)  # xyxy, (b, h*w, 4)
        
        lines_enc0 = torch.zeros((targets.shape[0], targets.shape[1], batch["lines_enc"].shape[1]), dtype=torch.long)        
        for i in range(batch_size):
            matches = batch_idx == i
            n = matches.sum()
            if n:
                lines_enc0[i,:n] = batch_enc[matches.cpu()]
        lines_enc0 = lines_enc0.to(self.device)
        
        boxless = batch['im_file'][0].split('/')[-1][0] == '4'
        boxless = False
        
        if not boxless:
            bboxes_for_assigner = pred_bboxes.clone().detach()
            # Angle don't need to be scaled
            bboxes_for_assigner[..., :4] *= stride_tensor

            mask_size = ((gt_bboxes[:,:,2:4].max(dim=2)[0]).unsqueeze(-1) <= (stride_tensor[:,0] * 2 * (self.reg_max - 1)))
            mask_size[:,:,stride_tensor[:,0] == stride_tensor.max()] = True
            _, target_bboxes, target_scores, fg_mask, _, lines_enc = self.assigner(
                pred_scores.detach().sigmoid(),
                bboxes_for_assigner.type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
                lines_enc0,
                mask_size
            )
            target_scores_sum = max(target_scores.sum(), 1)

            # Cls loss
            # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE        

            # Bbox loss
            target_bboxes[..., :4]
            if fg_mask.sum():
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes/stride_tensor, target_scores, target_scores_sum, fg_mask
                )
            else:
                loss[0] += (pred_angles * 0).sum()

            if self.true_boxes:
                pred_bboxes = gt_bboxes[:,:,:4].to(pred_bboxes.dtype)
                pred_angles = gt_bboxes[:,:,4].to(pred_angles.dtype)
                target_scores = mask_gt.to(self.device)
                fg_mask = mask_gt[...,0].to(self.device) > 0
                lines_enc = lines_enc0.to(self.device)
        
        if boxless:
            probs = feats[-1]
            target_scores,lines_enc,fg_mask,pairs = self.emulate_assigner(stride_tensor,lines_enc0,batch['lines'],batch['batch_idx'],\
                                                                          gt_labels,gt_bboxes,pred_bboxes,pred_scores,pred_angles,probs,None)
        
        if self.hyp.ctc > 0:
            val = torch.tensor(0.0).to(self.device)
            target_weights_sum = torch.tensor(0.0).to(self.device)
            for i in range(pred_scores.shape[0]):
                index = fg_mask[i]
                 
                pred_cells = pred_bboxes[i,index][:,:4] / stride_tensor[0]
                char_probs = scores_by_obb(pred_cells, pred_angles[i,index], feats[-1][i])
                char_probs = char_probs.to(self.last.end_conv.weight.dtype)
                char_probs = self.last(char_probs)
                char_probs = torch.nn.functional.log_softmax(char_probs.permute(1,0,2), dim=-1)
                
                ctc_loss = torch.nn.CTCLoss(blank=char_probs.shape[-1]-1, reduction="none")
                input_lengths = char_probs.shape[0] * torch.ones(char_probs.shape[1], dtype=torch.long).to(char_probs.device)
                target_enc = lines_enc[i,index]
                  
                target_lengths = (target_enc >= 0).sum(axis=-1)
                torch.cuda.empty_cache()
                enc_sh = self.charset['#'] if '#' in self.charset else -3
                enc_at = self.charset['@'] if '@' in self.charset else -2
                
                wc_mask = torch.logical_or((target_enc == enc_sh).any(dim = 1), (target_enc == enc_at).any(dim = 1))
                wc_mask = torch.logical_not(wc_mask)
                if wc_mask.any():
                    temp = ctc_loss(char_probs[:,wc_mask].float(), target_enc[wc_mask], input_lengths[wc_mask], target_lengths[wc_mask])
                    #temp = ctcloss_reference(char_probs[:,wc_mask].float(), target_enc[wc_mask], input_lengths[wc_mask], \
                    #                         target_lengths[wc_mask], blank=char_probs.shape[-1]-1, reduction="none")
                    target_weights = target_scores[i,index,0][wc_mask]
                    val += (temp * target_weights).sum()
                    target_weights_sum += target_weights.sum()
                  
            loss[3] = val / max(target_weights_sum, 1)        
        
        if boxless:
            if self.hyp.box > 0:
                val = 0
                target_weights_sum = 0
                for i in range(pred_scores.shape[0]):
                    idx1,idx2 = pairs[i][:,0],pairs[i][:,1]
                    obb1,obb2 = pred_bboxes[i,idx1],pred_bboxes[i,idx2]
                    obb1[:,:4] *= stride_tensor[idx1]
                    obb2[:,:4] *= stride_tensor[idx2]
                    temp = probiou(obb1,obb2)
                    score1,score2 = target_scores[i,idx1],target_scores[i,idx2]
                    val += ((1 - temp) * score1 * score2).sum()
                    target_weights_sum += (score1 * score2).sum()
                loss[0] = val / max(target_weights_sum, 1)  
        
            if self.hyp.cls > 0:
                loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / max(target_scores.sum(), 1)
        
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.ctc
        
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
