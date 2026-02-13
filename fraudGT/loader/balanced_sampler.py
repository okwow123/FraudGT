import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch_geometric.loader import LinkNeighborLoader
import numpy as np

from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_sampler


class BalancedEdgeSampler(Sampler):
    """
    클래스 밸런싱 엣지 샘플러
    자금세탁(positive) 샘플과 정상(negative) 샘플의 비율을 조정하여 학습
    """
    def __init__(self, edge_labels, pos_ratio=0.5, shuffle=True):
        """
        Args:
            edge_labels: [num_edges] tensor of 0/1 labels
            pos_ratio: positive 샘플 비율 (0.5 = 50% positive, 50% negative)
            shuffle: 셔플 여부
        """
        self.edge_labels = edge_labels
        self.pos_ratio = pos_ratio
        self.shuffle = shuffle

        # Positive와 Negative 인덱스 분리
        self.pos_indices = torch.where(edge_labels == 1)[0].cpu().numpy()
        self.neg_indices = torch.where(edge_labels == 0)[0].cpu().numpy()

        self.num_pos = len(self.pos_indices)
        self.num_neg = len(self.neg_indices)

        # 전체 샘플 수 계산 (positive 수에 기반)
        self.num_samples = int(self.num_pos / pos_ratio)

    def __iter__(self):
        # Positive 샘플 전부 사용
        if self.shuffle:
            pos_samples = np.random.choice(self.pos_indices, size=self.num_pos, replace=False)
        else:
            pos_samples = self.pos_indices

        # Negative 샘플은 비율에 맞춰 샘플링
        num_neg_samples = self.num_samples - self.num_pos
        num_neg_samples = min(num_neg_samples, self.num_neg)  # 최대 negative 수 제한

        if self.shuffle:
            neg_samples = np.random.choice(self.neg_indices, size=num_neg_samples, replace=False)
        else:
            neg_samples = self.neg_indices[:num_neg_samples]

        # Positive와 Negative 합치기
        all_samples = np.concatenate([pos_samples, neg_samples])

        # 셔플
        if self.shuffle:
            np.random.shuffle(all_samples)

        return iter(all_samples.tolist())

    def __len__(self):
        return self.num_samples


@register_sampler('balanced_link_neighbor')
def create_balanced_link_neighbor_loader(dataset, split):
    """
    밸런싱된 LinkNeighborLoader 생성
    """
    data = dataset[split]
    task_entity = cfg.dataset.task_entity

    # 엣지 레이블 가져오기
    edge_labels = data[task_entity].y

    # 밸런싱 샘플러 생성
    if split == 'train':
        # 학습 시에는 클래스 밸런싱 적용 (positive 30-40%)
        pos_ratio = cfg.train.pos_ratio if hasattr(cfg.train, 'pos_ratio') else 0.35
        sampler = BalancedEdgeSampler(edge_labels, pos_ratio=pos_ratio, shuffle=True)
    else:
        # Validation/Test는 전체 데이터 사용
        sampler = None

    # LinkNeighborLoader 생성
    loader = LinkNeighborLoader(
        data,
        num_neighbors=cfg.train.neighbor_sizes if split == 'train' else cfg.val.neighbor_sizes if hasattr(cfg.val, 'neighbor_sizes') else cfg.train.neighbor_sizes,
        edge_label_index=data[task_entity].edge_index,
        edge_label=edge_labels,
        batch_size=cfg.train.batch_size if split == 'train' else cfg.val.batch_size if hasattr(cfg.val, 'batch_size') else cfg.train.batch_size,
        shuffle=(split == 'train') and (sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers if split == 'train' else 0,
        persistent_workers=cfg.train.persistent_workers if split == 'train' and cfg.num_workers > 0 else False,
        pin_memory=cfg.train.pin_memory if split == 'train' else False,
    )

    return loader


def compute_class_weights(dataset):
    """
    데이터셋의 클래스 분포를 기반으로 가중치 계산
    """
    train_data = dataset['train']
    task_entity = cfg.dataset.task_entity

    labels = train_data[task_entity].y
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()
    total = num_pos + num_neg

    # Inverse frequency weighting
    weight_pos = total / (2 * num_pos) if num_pos > 0 else 1.0
    weight_neg = total / (2 * num_neg) if num_neg > 0 else 1.0

    # Normalize
    weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32)
    weights = weights / weights.sum() * 2  # Scale to sum to 2

    print(f"Class distribution - Negative: {num_neg}, Positive: {num_pos}")
    print(f"Class weights - Negative: {weight_neg:.4f}, Positive: {weight_pos:.4f}")

    return weights.tolist()
