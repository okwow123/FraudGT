import logging
import time

import torch

from fraudGT.graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loss import compute_loss
from fraudGT.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))

        # === [NEW] apply masking only in training ============================
        node_ratio = _get_ratio(0.0, 'data.node_mask_ratio')
        edge_ratio = _get_ratio(0.15, 'data.edge_mask_ratio')
        if hasattr(batch, 'x'):
            batch.x = mask_node_features(batch.x, node_ratio)
        if hasattr(batch, 'edge_attr'):
            batch.edge_attr = mask_edge_features(batch.edge_attr, edge_ratio)
        # =====================================================================

        
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def train(loggers, loaders, model, optimizer, scheduler):
    r"""
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))

# === [NEW] masking utilities ================================================
def _get_ratio(default_val: float, path: str):
    """cfg.data.edge_mask_ratio / cfg.data.node_mask_ratio 안전 접근."""
    try:
        # cfg는 dotdict라 getattr 체인으로 안전 접근
        d = getattr(cfg, 'data', None)
        if d is None:
            return default_val
        return float(getattr(d, path.split('.')[-1], default_val))
    except Exception:
        return default_val

def mask_edge_features(edge_attr, mask_ratio: float):
    if edge_attr is None or mask_ratio <= 0.0:
        return edge_attr
    E = edge_attr.size(0)
    mask = torch.rand(E, device=edge_attr.device) < mask_ratio
    out = edge_attr.clone()
    out[mask] = 0
    return out

def mask_node_features(x, mask_ratio: float):
    if x is None or mask_ratio <= 0.0:
        return x
    N = x.size(0)
    mask = torch.rand(N, device=x.device) < mask_ratio
    out = x.clone()
    out[mask] = 0
    return out
# ============================================================================
