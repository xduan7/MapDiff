import torch
import torch.nn.functional as F
import math
from torch import nn
from model.ipa.rigid_utils import Rigid
from model.ipa.ipa_utils import cal_dihedrals, cal_pair_rbf, relative_pairwise_position_idx, LayerNorm
from model.ipa.ipa_attn import InvariantPointAttention, StructureModuleTransition, EdgeTransition
from einops import rearrange


class NodeMaskEncoder(nn.Module):
    def __init__(self, emb_dim, num_d_feat=6, num_aa_types=20):
        super(NodeMaskEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.aa_in_proj = nn.Linear(num_aa_types, emb_dim)
        self.d_in_proj = nn.Linear(num_d_feat, emb_dim)
        self.mask_emb = nn.Embedding(1, emb_dim)
        self.pad_emb = nn.Embedding(1, emb_dim, padding_idx=0)
        self.pe_encoding = nn.Parameter(self.positional_encoding(max_len=1200), requires_grad=False)

        torch.nn.init.xavier_uniform_(self.aa_in_proj.weight.data)
        torch.nn.init.xavier_uniform_(self.d_in_proj.weight.data)

    def forward(self, x_aa, x_pos, node_mask, seq_mask):
        x = self.aa_in_proj(x_aa)
        x[node_mask == 1] = self.mask_emb.weight
        x[seq_mask == 0] = self.pad_emb.weight
        d_feat = cal_dihedrals(x_pos).float()
        x_d = self.d_in_proj(d_feat)
        x = x + x_d + self.pe_encoding[:, :x.size(1), :]
        return x

    def positional_encoding(self, max_len=1200):
        pe = torch.zeros(max_len, self.emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2).float() * (-math.log(10000.0) / self.emb_dim))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        return pe


class EdgePairEncoder(nn.Module):
    def __init__(self, emb_dim, dist_bins=24, dist_bin_width=0.5, rel_pos_k=32):
        super(EdgePairEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.dist_bins = dist_bins
        self.dist_bin_width = dist_bin_width
        self.rel_pos_k = rel_pos_k
        self.rbf_in_proj = nn.Linear(self.dist_bins * 16, emb_dim)
        self.relpos_in_proj = nn.Linear(2 * self.rel_pos_k + 1, emb_dim)

        torch.nn.init.xavier_uniform_(self.rbf_in_proj.weight.data)
        torch.nn.init.xavier_uniform_(self.relpos_in_proj.weight.data)

    def forward(self, x_pos):
        batch_size, seq_len = x_pos.shape[:2]
        # pairwise distance rbf embedding
        x_bb = x_pos[:, :, :4]
        x_bb = rearrange(x_bb, 'b n c d -> b (n c) d', c=4)
        pairwise_distance = torch.cdist(x_bb, x_bb)
        pairwise_rbf = cal_pair_rbf(pairwise_distance, self.dist_bins, self.dist_bin_width)
        pairwise_rbf = rearrange(pairwise_rbf, "b (n1 c1) (n2 c2) d -> b n1 n2 (c1 c2 d)", c1=4, c2=4)
        z = self.rbf_in_proj(pairwise_rbf)
        # relative position embedding
        rel_pos = relative_pairwise_position_idx(seq_len, self.rel_pos_k)
        rel_pos = F.one_hot(rel_pos, 2 * self.rel_pos_k + 1).float()
        rel_pos = rel_pos.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(x_pos.device)
        z = z + self.relpos_in_proj(rel_pos)
        return z


class IPANetModel(nn.Module):
    def __init__(self, ipa_dim=128, ipa_pairwise_dim=128, ipa_heads=4, ipa_depth=8, ipa_qk_points=4, ipa_v_points=8, dropout_rate=0.1):
        super(IPANetModel, self).__init__()
        self.ipa_layers = nn.ModuleList([])
        for i in range(ipa_depth):
            ipa = InvariantPointAttention(ipa_dim, ipa_pairwise_dim, ipa_dim // ipa_heads, ipa_heads, ipa_qk_points,
                                          ipa_v_points)
            ipa_dropout = nn.Dropout(dropout_rate)
            layer_norm_ipa = LayerNorm(ipa_dim)
            pre_transit = nn.Linear(ipa_dim, ipa_dim * 4)
            post_transit = nn.Linear(ipa_dim * 4, ipa_dim)
            transition = StructureModuleTransition(ipa_dim * 4, 1, 0.1)
            if i == ipa_depth - 1:
                edge_transition = None
            else:
                edge_transition = EdgeTransition(ipa_dim, ipa_pairwise_dim, ipa_pairwise_dim, num_layers=2)
            self.ipa_layers.append(nn.ModuleList([ipa, ipa_dropout, layer_norm_ipa, pre_transit, transition,
                                                  post_transit, edge_transition]))

    def forward(self, s, z, r, seq_mask, attn_drop_rate):
        for ipa, ipa_dropout, layer_norm_ipa, *transit_layers, edge_transition in self.ipa_layers:
            s = s + ipa(s, z, r, seq_mask, attn_drop_rate=attn_drop_rate)
            s = ipa_dropout(s)
            s = layer_norm_ipa(s)

            pre_transit = transit_layers[0]
            transition = transit_layers[1]
            post_transit = transit_layers[2]
            s = pre_transit(s)
            s = transition(s)
            s = post_transit(s)

            if edge_transition is not None:
                z = edge_transition(s, z)
                # z = checkpoint(edge_transition, s, z)
        return s


class IPANetPredictor(nn.Module):
    def __init__(self, dropout=0.1, hidden_dim=128, ipa_dim=128, ipa_pairwise_dim=128, ipa_heads=4, ipa_depth=6,
                 ipa_qk_points=4, ipa_v_points=8):
        super(IPANetPredictor, self).__init__()
        self.node_encoder = NodeMaskEncoder(hidden_dim)
        self.edge_pair_encoder = EdgePairEncoder(hidden_dim)
        self.s_dropout = nn.Dropout(dropout)
        self.z_dropout = nn.Dropout(dropout)
        self.node_predictor = nn.Linear(hidden_dim, 20)

        self.ipa = IPANetModel(ipa_dim, ipa_pairwise_dim, ipa_heads, ipa_depth, ipa_qk_points, ipa_v_points, dropout)

    def forward(self, x_aa, x_pos, x_aa_mask, seq_mask):
        r = Rigid.from_3_points(x_pos[:, :, 0], x_pos[:, :, 1], x_pos[:, :, 2])
        s = self.node_encoder(x_aa, x_pos, x_aa_mask, seq_mask)
        z = self.edge_pair_encoder(x_pos)
        s = self.s_dropout(s)
        z = self.z_dropout(z)
        seq_mask = seq_mask.long()

        s = self.ipa(s, z, r, seq_mask, attn_drop_rate=0.0)
        logit = self.node_predictor(s)

        return logit
