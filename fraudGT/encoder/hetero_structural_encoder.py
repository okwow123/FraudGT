import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
import numpy as np

from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_node_encoder


def compute_graph_structural_features(batch):
    """
    자금세탁 탐지를 위한 고급 그래프 구조적 특징 계산
    """
    edge_index = batch[('node', 'to', 'node')].edge_index
    num_nodes = batch['node'].num_nodes

    features = []

    # 1. In-degree & Out-degree (계좌로 들어오고 나가는 거래 수)
    in_deg = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float)
    out_deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    features.extend([in_deg.unsqueeze(1), out_deg.unsqueeze(1)])

    # 2. Total degree
    total_deg = in_deg + out_deg
    features.append(total_deg.unsqueeze(1))

    # 3. Degree ratio (in/out 비율 - 자금세탁은 불균형적)
    deg_ratio = torch.log((in_deg + 1.0) / (out_deg + 1.0))
    features.append(deg_ratio.unsqueeze(1))

    # 4. Degree centrality (normalized)
    deg_centrality = total_deg / (num_nodes - 1 + 1e-6)
    features.append(deg_centrality.unsqueeze(1))

    # 5. Unique neighbor counts
    unique_in = torch.zeros(num_nodes, device=edge_index.device)
    unique_out = torch.zeros(num_nodes, device=edge_index.device)

    for i in range(num_nodes):
        mask_in = edge_index[1] == i
        mask_out = edge_index[0] == i
        if mask_in.sum() > 0:
            unique_in[i] = edge_index[0][mask_in].unique().size(0)
        if mask_out.sum() > 0:
            unique_out[i] = edge_index[1][mask_out].unique().size(0)

    features.extend([unique_in.unsqueeze(1), unique_out.unsqueeze(1)])

    # 6. Average neighbor degree (이웃 노드들의 평균 degree)
    avg_neighbor_deg = torch.zeros(num_nodes, device=edge_index.device)
    for i in range(num_nodes):
        mask = edge_index[0] == i
        if mask.sum() > 0:
            neighbors = edge_index[1][mask]
            avg_neighbor_deg[i] = total_deg[neighbors].mean()
    features.append(avg_neighbor_deg.unsqueeze(1))

    # 7. Clustering coefficient (approximation for directed graphs)
    clustering_coef = torch.zeros(num_nodes, device=edge_index.device)
    for i in range(num_nodes):
        # Get neighbors
        out_mask = edge_index[0] == i
        neighbors = edge_index[1][out_mask].unique()

        k = neighbors.size(0)
        if k > 1:
            # Count triangles (simplified)
            edge_count = 0
            for j in neighbors:
                for l in neighbors:
                    if j != l:
                        # Check if edge exists between j and l
                        edge_exists = ((edge_index[0] == j) & (edge_index[1] == l)).any()
                        if edge_exists:
                            edge_count += 1

            clustering_coef[i] = edge_count / (k * (k - 1) + 1e-6)

    features.append(clustering_coef.unsqueeze(1))

    # 8. Hub score (nodes with high out-degree are hubs)
    hub_score = out_deg / (total_deg.max() + 1e-6)
    features.append(hub_score.unsqueeze(1))

    # 9. Authority score (nodes with high in-degree are authorities)
    authority_score = in_deg / (total_deg.max() + 1e-6)
    features.append(authority_score.unsqueeze(1))

    # Concatenate all features
    structural_features = torch.cat(features, dim=1)

    # Normalize features
    mean = structural_features.mean(dim=0, keepdim=True)
    std = structural_features.std(dim=0, keepdim=True) + 1e-6
    structural_features = (structural_features - mean) / std

    return structural_features


@register_node_encoder('Hetero_Raw_Structural')
class HeteroRawStructuralEncoder(nn.Module):
    """
    원본 특징 + 그래프 구조 특징을 결합한 고급 인코더
    자금세탁 탐지를 위한 네트워크 토폴로지 정보 활용
    """
    def __init__(self, dim_emb, dataset):
        super().__init__()

        # Import base encoder
        from fraudGT.encoder.hetero_raw_encoder import HeteroRawNodeEncoder

        # Base encoder for raw features
        self.raw_encoder = HeteroRawNodeEncoder(dim_emb, dataset)

        # Structural features dimension
        # (in_deg, out_deg, total_deg, deg_ratio, deg_centrality,
        #  unique_in, unique_out, avg_neighbor_deg, clustering, hub, authority)
        self.structural_dim = 11

        # Project structural features to embedding space
        self.structural_proj = nn.Sequential(
            nn.Linear(self.structural_dim, dim_emb // 2),
            nn.LayerNorm(dim_emb // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_emb // 2, dim_emb // 4)
        )

        # Combine raw and structural features
        if not isinstance(dim_emb, dict):
            combined_dim = dim_emb + dim_emb // 4
        else:
            # For hetero case, use the node type embedding
            combined_dim = dim_emb['node'] + dim_emb['node'] // 4 if 'node' in dim_emb else dim_emb + dim_emb // 4

        self.final_proj = nn.Sequential(
            nn.Linear(combined_dim, dim_emb if not isinstance(dim_emb, dict) else dim_emb['node'] if 'node' in dim_emb else dim_emb),
            nn.LayerNorm(dim_emb if not isinstance(dim_emb, dict) else dim_emb['node'] if 'node' in dim_emb else dim_emb),
        )

        self.dim_emb = dim_emb

    def forward(self, batch):
        # Encode raw features
        batch = self.raw_encoder(batch)

        # Compute structural features
        structural_feats = compute_graph_structural_features(batch)
        structural_emb = self.structural_proj(structural_feats)

        # Combine features for node type
        if isinstance(batch, HeteroData):
            for node_type in batch.node_types:
                if hasattr(batch[node_type], 'x'):
                    # Concatenate raw and structural embeddings
                    combined = torch.cat([batch[node_type].x, structural_emb], dim=1)
                    batch[node_type].x = self.final_proj(combined)
        else:
            # Homogeneous case
            combined = torch.cat([batch.x, structural_emb], dim=1)
            batch.x = self.final_proj(combined)

        return batch
