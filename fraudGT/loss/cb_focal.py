# fraudGT/loss/cb_focal.py
import torch
import torch.nn.functional as F
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_loss

@register_loss('cb_focal')
def cb_focal_loss(pred, true, epoch):
    """Class-Balanced Focal Loss (binary 또는 multiclass).
    사용법 (cfg.model 하위):
      - loss_fun: 'cb_focal'
      - cb_beta: float in (0,1), default 0.99
      - cb_gamma: focal gamma, default 1.5
      - loss_class_counts: [c0, c1, ...] (옵션, 전체 학습셋 클래스별 개수)
    """
    beta = getattr(cfg.model, 'cb_beta', 0.99)
    gamma = getattr(cfg.model, 'cb_gamma', 1.5)
    device = pred.device

    # shape 정리
    pred = pred.squeeze(-1) if pred.ndim > 1 and pred.shape[-1] == 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 and true.shape[-1] == 1 else true

    # ---- Binary case ----
    if pred.ndim == 1 or (pred.ndim == 2 and pred.shape[1] == 1):
        num_classes = 2
        counts = getattr(cfg.model, 'loss_class_counts', [])
        if counts:
            counts = torch.tensor(counts, dtype=torch.float32, device=device)
            if counts.numel() != 2:
                counts = None
        if not counts:
            # 학습셋 전역 카운트가 없으면, 배치 분포로 대체 (근사)
            counts = torch.bincount(true.long(), minlength=2).to(device=device, dtype=torch.float32)

        effective_num = 1.0 - torch.pow(torch.tensor(beta, device=device), counts)
        weights = (1.0 - beta) / effective_num
        weights = weights * (num_classes / weights.sum())  # 합을 C로 정규화

        labels = true.float()
        p = torch.sigmoid(pred)
        p_t = p * labels + (1 - p) * (1 - labels)
        bce = F.binary_cross_entropy_with_logits(pred, labels, reduction='none')
        focal = torch.pow(1 - p_t, gamma)
        alpha_t = weights[true.long()]               # 클래스별 가중치 매핑
        loss = alpha_t * focal * bce
        return loss.mean(), torch.sigmoid(pred)

    # ---- Multiclass case ----
    N, C = pred.shape
    counts = getattr(cfg.model, 'loss_class_counts', [])
    if counts:
        counts = torch.tensor(counts, dtype=torch.float32, device=device)
        if counts.numel() != C:
            counts = None
    if not counts:
        counts = torch.bincount(true.long(), minlength=C).to(device=device, dtype=torch.float32)

    effective_num = 1.0 - torch.pow(torch.tensor(beta, device=device), counts)
    weights = (1.0 - beta) / effective_num
    weights = weights * (C / weights.sum())

    labels = F.one_hot(true.long(), num_classes=C).float().to(device)
    p = torch.softmax(pred, dim=1)
    p_t = (p * labels).sum(dim=1)
    ce = F.cross_entropy(pred, true.long(), reduction='none')
    focal = torch.pow(1 - p_t, gamma)
    alpha_t = weights[true.long()]
    loss = alpha_t * focal * ce
    return loss.mean(), p
