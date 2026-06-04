# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonality_loss(z_anat, z_patho):
    """Force anatomy and pathology vectors to be uncorrelated."""
    # z_anat:  (B, 256) after pooling
    # z_patho: (B, 64)
    # Normalize both, compute cross-covariance, penalize its norm
    za = F.normalize(z_anat, dim=-1)
    zp = F.normalize(z_patho, dim=-1)
    cross = torch.mm(za.T, zp)           # (256, 64)
    return cross.pow(2).mean()


def dice_loss(pred_logits, target, smooth=1.0):
    """
    pred_logits: (B, 3, H, W)
    target:      (B, H, W) long  — 0=bg, 1=lung, 2=heart
    """
    pred = F.softmax(pred_logits, dim=1)
    target_oh = F.one_hot(target, num_classes=3).permute(0,3,1,2).float()
    num = (2 * pred * target_oh).sum(dim=(2,3)) + smooth
    den = (pred + target_oh).sum(dim=(2,3)) + smooth
    return 1 - (num / den).mean()


class DisentanglementLoss(nn.Module):
    def __init__(self, lambda_ortho=1.0, lambda_seg=5.0, lambda_cls=0.5):
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.lambda_seg   = lambda_seg
        self.lambda_cls   = lambda_cls
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z_spatial, z_patho, seg_logits,
                seg_masks, patho_labels, patho_classifier):
        """
        z_spatial:        (B, 256, 32, 32)
        z_patho:          (B, 64)
        seg_logits:       (B, 3, 512, 512)
        seg_masks:        (B, 512, 512) long  — pseudo or real masks
        patho_labels:     (B,) long  0/1
        patho_classifier: small MLP trained on z_anatomy to predict disease
                          (we MAXIMIZE its loss from anatomy encoder's perspective)
        """
        # 1. Segmentation supervision
        l_seg = dice_loss(seg_logits, seg_masks)

        # 2. Orthogonality between anatomy (pooled) and pathology
        z_anat_pooled = z_spatial.mean(dim=(2,3))   # (B, 256)
        l_ortho = orthogonality_loss(z_anat_pooled, z_patho)

        # 3. Adversarial: anatomy should NOT be able to predict disease
        #    We maximize CE → anatomy encoder gradient pushed away from disease info
        disease_pred = patho_classifier(z_anat_pooled)
        l_adv = -self.ce(disease_pred, patho_labels)   # negative = maximise

        total = (self.lambda_seg   * l_seg
               + self.lambda_ortho * l_ortho
               + self.lambda_cls   * l_adv)

        return total, {"seg": l_seg.item(),
                       "ortho": l_ortho.item(),
                       "adv": l_adv.item()}