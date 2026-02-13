import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index

from fraudGT.graphgym.register import register_head
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.layer import MLP


class MultiHeadAttentionEdgeDecoder(nn.Module):
    """
    Multi-Head Attention 기반 엣지 디코딩
    송금인, 수취인, 거래 특징 간의 중요도를 동적으로 학습
    """
    def __init__(self, dim_in, dim_out, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_in // num_heads
        assert dim_in % num_heads == 0, f"dim_in ({dim_in}) must be divisible by num_heads ({num_heads})"

        # Separate projections for source, target, edge
        self.src_proj = nn.Linear(dim_in, dim_in)
        self.dst_proj = nn.Linear(dim_in, dim_in)
        self.edge_proj = nn.Linear(dim_in, dim_in)

        # Attention mechanism
        self.query_proj = nn.Linear(dim_in, dim_in)
        self.key_proj = nn.Linear(dim_in, dim_in)
        self.value_proj = nn.Linear(dim_in, dim_in)

        # Output projection
        self.output_proj = nn.Linear(dim_in, dim_in)

        # Final MLP for classification
        self.classifier = MLP(
            dim_in * 2,  # Concatenate attended features + original features
            dim_out,
            num_layers=3,
            bias=True
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_in)

    def forward(self, src_emb, dst_emb, edge_attr):
        """
        Args:
            src_emb: [batch, dim_in] 송금인 임베딩
            dst_emb: [batch, dim_in] 수취인 임베딩
            edge_attr: [batch, dim_in] 엣지 속성 (거래 정보)

        Returns:
            [batch, dim_out] 예측 결과
        """
        batch_size = src_emb.size(0)

        # Project features
        src = self.src_proj(src_emb)
        dst = self.dst_proj(dst_emb)
        edge = self.edge_proj(edge_attr)

        # Combine into sequence: [batch, 3, dim_in]
        # Sequence: [source, destination, edge]
        seq = torch.stack([src, dst, edge], dim=1)

        # Multi-head attention
        # Reshape to [batch, 3, num_heads, dim_head]
        query = self.query_proj(seq).view(batch_size, 3, self.num_heads, self.dim_head).transpose(1, 2)
        key = self.key_proj(seq).view(batch_size, 3, self.num_heads, self.dim_head).transpose(1, 2)
        value = self.value_proj(seq).view(batch_size, 3, self.num_heads, self.dim_head).transpose(1, 2)

        # Compute attention scores: [batch, num_heads, 3, 3]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention: [batch, num_heads, 3, dim_head]
        attended = torch.matmul(attn_weights, value)

        # Reshape back: [batch, 3, dim_in]
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 3, -1)

        # Output projection
        attended = self.output_proj(attended)

        # Residual connection + Layer norm
        attended = self.layer_norm(attended + seq)

        # Aggregate across sequence dimension (take mean)
        aggregated = attended.mean(dim=1)  # [batch, dim_in]

        # Concatenate with original edge context (source-target interaction)
        edge_context = torch.cat([src_emb, dst_emb], dim=-1)  # [batch, dim_in*2]

        # Final classification
        output = self.classifier(edge_context)

        return output


class BilinearEdgeDecoder(nn.Module):
    """
    Bilinear 기반 엣지 디코딩 (대안)
    송금인과 수취인 간의 학습 가능한 상호작용 행렬
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()

        # Bilinear transformation
        self.bilinear = nn.Bilinear(dim_in, dim_in, dim_in)

        # Edge feature integration
        self.edge_transform = nn.Linear(dim_in, dim_in)

        # Final classifier
        self.classifier = MLP(
            dim_in * 3,  # bilinear + source + target
            dim_out,
            num_layers=3,
            bias=True
        )

    def forward(self, src_emb, dst_emb, edge_attr):
        """
        Args:
            src_emb: [batch, dim_in]
            dst_emb: [batch, dim_in]
            edge_attr: [batch, dim_in]

        Returns:
            [batch, dim_out]
        """
        # Bilinear interaction between source and destination
        interaction = self.bilinear(src_emb, dst_emb)

        # Transform edge features
        edge_transformed = self.edge_transform(edge_attr)

        # Combine all features
        combined = torch.cat([interaction, edge_transformed, src_emb * dst_emb], dim=-1)

        # Classify
        output = self.classifier(combined)

        return output


@register_head('hetero_edge_attention')
class HeteroGNNEdgeHeadAttention(nn.Module):
    """
    Multi-Head Attention 기반 Hetero GNN Edge Head
    자금세탁 탐지를 위한 고급 엣지 디코딩
    """
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        # Split indices
        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        # Choose decoder type
        decoder_type = cfg.model.edge_decoder_type if hasattr(cfg.model, 'edge_decoder_type') else 'attention'

        if decoder_type == 'attention':
            num_heads = cfg.model.edge_attn_heads if hasattr(cfg.model, 'edge_attn_heads') else 8
            self.decoder = MultiHeadAttentionEdgeDecoder(dim_in, dim_out, num_heads=num_heads)
        elif decoder_type == 'bilinear':
            self.decoder = BilinearEdgeDecoder(dim_in, dim_out)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def _apply_index(self, batch):
        """
        배치에서 해당 split의 엣지만 추출
        """
        task = cfg.dataset.task_entity

        # Filter edges for current split
        mask = torch.isin(batch[task].e_id,
                          getattr(self, f'{batch.split}_inds')[batch[task].input_id])

        edge_index = batch[task].edge_index

        # Extract node embeddings and edge attributes
        src_emb = batch[task[0]].x[edge_index[0, mask]]
        dst_emb = batch[task[2]].x[edge_index[1, mask]]
        edge_attr = batch[task].edge_attr[mask]

        label = batch[task].y[mask]

        return src_emb, dst_emb, edge_attr, label

    def forward(self, batch):
        """
        Forward pass
        """
        src_emb, dst_emb, edge_attr, label = self._apply_index(batch)

        # Decode edge predictions
        pred = self.decoder(src_emb, dst_emb, edge_attr)

        return pred, label


@register_head('hetero_edge_bilinear')
class HeteroGNNEdgeHeadBilinear(nn.Module):
    """
    Bilinear 기반 Hetero GNN Edge Head
    """
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        self.decoder = BilinearEdgeDecoder(dim_in, dim_out)

    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        mask = torch.isin(batch[task].e_id,
                          getattr(self, f'{batch.split}_inds')[batch[task].input_id])

        edge_index = batch[task].edge_index

        src_emb = batch[task[0]].x[edge_index[0, mask]]
        dst_emb = batch[task[2]].x[edge_index[1, mask]]
        edge_attr = batch[task].edge_attr[mask]
        label = batch[task].y[mask]

        return src_emb, dst_emb, edge_attr, label

    def forward(self, batch):
        src_emb, dst_emb, edge_attr, label = self._apply_index(batch)
        pred = self.decoder(src_emb, dst_emb, edge_attr)
        return pred, label
