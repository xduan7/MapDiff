import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.nn import Linear
from .egnn_pyg import EGNN_Sparse
from .utils import NodeMaskEncoder, edgeEncoder


class MASK_EGNN_NET(torch.nn.Module):
    def __init__(self, input_feat_dim, hidden_channels, edge_attr_dim, dropout=0.0, n_layers=1, output_dim=20,
                 embedding=False, embedding_dim=64, mlp_num=2, update_edge=True, embed_ss=-1, norm_feat=False):
        super(MASK_EGNN_NET, self).__init__()
        self.dropout = dropout

        self.update_edge = update_edge
        self.mpnn_layes = nn.ModuleList()
        self.ff_list = nn.ModuleList()

        self.embedding = embedding
        self.embed_ss = embed_ss
        self.n_layers = n_layers
        if embedding:
            self.ss_mlp = nn.Sequential(nn.Linear(8, hidden_channels), nn.SiLU(),
                                        nn.Linear(hidden_channels, embedding_dim))
        else:
            self.ss_mlp = nn.Sequential(nn.Linear(8, hidden_channels), nn.SiLU(),
                                        nn.Linear(hidden_channels, input_feat_dim))

        for i in range(n_layers):
            if embedding:
                layer = EGNN_Sparse(embedding_dim, m_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    mlp_num=mlp_num, update_edge=self.update_edge, norm_feats=norm_feat)
            else:
                layer = EGNN_Sparse(input_feat_dim, m_dim=hidden_channels, edge_attr_dim=edge_attr_dim, dropout=dropout,
                                    mlp_num=mlp_num, update_edge=self.update_edge, norm_feats=norm_feat)
            self.mpnn_layes.append(layer)

            if embedding:
                ff_layer = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Dropout(p=dropout), nn.SiLU(),
                                         torch_geometric.nn.norm.LayerNorm(embedding_dim),
                                         nn.Linear(embedding_dim, embedding_dim))
            else:
                ff_layer = nn.Sequential(nn.Linear(input_feat_dim, input_feat_dim), nn.Dropout(p=dropout), nn.SiLU(),
                                         torch_geometric.nn.norm.LayerNorm(input_feat_dim),
                                         nn.Linear(input_feat_dim, input_feat_dim))

            self.ff_list.append(ff_layer)

        if embedding:
            self.node_embedding = NodeMaskEncoder(embedding_dim)
            self.edge_embedding = edgeEncoder(embedding_dim)
            self.lin = Linear(embedding_dim, output_dim)
        else:
            self.lin = Linear(input_feat_dim, output_dim)

    def forward(self, data):
        # data.x first 20 dim is noise label. 21 to 34 is knowledge from backbone, e.g. mu_r_norm, sasa, b factor and so on

        x, pos, extra_x, edge_index, edge_attr, ss, batch, mask_index = data.x, data.pos, data.extra_x, data.edge_index, data.edge_attr, data.ss, data.batch, data.mask_index

        ss_embed = self.ss_mlp(ss)

        x = torch.cat([x, extra_x], dim=1)
        if self.embedding:
            x = self.node_embedding(x, mask_index)
            edge_attr = self.edge_embedding(edge_attr)

        if self.embed_ss == -3:
            x = x + ss_embed

        x = torch.cat([pos, x], dim=1)

        for i, layer in enumerate(self.mpnn_layes):

            if self.embed_ss == -2 and i == self.n_layers - 1:
                corr, feats = x[:, 0:3], x[:, 3:]
                feats = feats + ss_embed  # [N,hidden_dim]+[N,hidden_dim]
                x = torch.cat([corr, feats], dim=-1)

            if self.update_edge:
                h, edge_attr = layer(x, edge_index, edge_attr, batch)  # [N,hidden_dim]
            else:
                h = layer(x, edge_index, edge_attr, batch)  # [N,hidden_dim]

            corr, feats = h[:, 0:3], h[:, 3:]
            feats = self.ff_list[i](feats)

            x = torch.cat([corr, feats], dim=-1)

        corr, x = x[:, 0:3], x[:, 3:]

        if self.embed_ss == -1:
            x = x + ss_embed

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x
