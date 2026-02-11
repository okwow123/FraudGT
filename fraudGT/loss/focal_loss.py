import torch
import torch.nn.functional as F
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_loss


@register_loss('focal_loss')
def focal_loss(pred, true, epoch):
    """
    Focal Loss for imbalanced fraud detection
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma > 0: down-weight easy examples
    alpha: balance positive/negative examples
    """
    gamma = cfg.model.focal_gamma if hasattr(cfg.model, 'focal_gamma') else 2.0
    alpha = cfg.model.focal_alpha if hasattr(cfg.model, 'focal_alpha') else 0.75
    
    # Binary classification
    if pred.ndim == 1:
        bce_loss = F.binary_cross_entropy_with_logits(pred, true.float(), reduction='none')
        p_t = torch.exp(-bce_loss)  # p for true class
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** gamma
        
        # Alpha balancing
        alpha_t = true * alpha + (1 - true) * (1 - alpha)
        
        loss = alpha_t * focal_term * bce_loss
        return loss.mean(), torch.sigmoid(pred)
    
    # Multiclass
    else:
        pred_softmax = F.softmax(pred, dim=-1)
        pred_log = F.log_softmax(pred, dim=-1)
        
        # Get probability of true class
        true_class_prob = pred_softmax.gather(1, true.unsqueeze(1)).squeeze(1)
        
        # Focal term
        focal_term = (1 - true_class_prob) ** gamma
        
        # Cross entropy
        ce_loss = F.nll_loss(pred_log, true, reduction='none')
        
        loss = focal_term * ce_loss
        return loss.mean(), pred_softmax


@register_loss('focal_loss_with_hard_mining')
def focal_loss_with_hard_mining(pred, true, epoch):
    """
    Focal Loss + Hard Negative Mining
    자금세탁과 유사하지만 실제로는 정상 거래인 어려운 샘플에 집중
    """
    gamma = 2.0
    alpha = 0.75
    hard_ratio = 0.3  # Top 30% 어려운 샘플에 집중
    
    if pred.ndim == 1:
        bce_loss = F.binary_cross_entropy_with_logits(pred, true.float(), reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** gamma
        alpha_t = true * alpha + (1 - true) * (1 - alpha)
        
        loss_per_sample = alpha_t * focal_term * bce_loss
        
        # Hard Negative Mining: 어려운 negative 샘플 선택
        num_hard = int(hard_ratio * (true == 0).sum())
        if num_hard > 0:
            neg_losses = loss_per_sample[true == 0]
            hard_negatives = torch.topk(neg_losses, num_hard)[0]
            
            # Positive는 모두, Negative는 hard만
            final_loss = torch.cat([loss_per_sample[true == 1], hard_negatives])
        else:
            final_loss = loss_per_sample
            
        return final_loss.mean(), torch.sigmoid(pred)
    else:
        # Multiclass는 기본 focal loss
        return focal_loss(pred, true, epoch)
