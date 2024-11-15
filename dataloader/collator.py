import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch, Data


class CollatorPretrain(object):
    def __init__(self, candi_rate=0.7, mask_rate=0.8, replace_rate=0.1, keep_rate=0.1):
        self.candi_rate = candi_rate
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.keep_rate = keep_rate

    def __call__(self, g_lst):
        g_batch = Batch.from_data_list(g_lst)
        self.random_mask_nodes(g_batch)
        return g_batch

    def random_mask_nodes(self, g):
        num_nodes = g.num_nodes
        all_node_ids = np.arange(num_nodes, dtype=np.int64)
        candi_ids = np.random.choice(all_node_ids, size=int(num_nodes * self.candi_rate), replace=False)
        mask_ids = np.random.choice(candi_ids, size=int(len(candi_ids) * self.mask_rate), replace=False)

        candi_ids = np.setdiff1d(candi_ids, mask_ids)
        replace_ids = np.random.choice(candi_ids, size=int(
            len(candi_ids) * (self.replace_rate / (self.replace_rate + self.keep_rate))), replace=False)
        keep_ids = np.setdiff1d(candi_ids, replace_ids)

        mask_index = torch.zeros(num_nodes, dtype=torch.long)
        mask_index[mask_ids] = 1
        mask_index[replace_ids] = 2
        mask_index[keep_ids] = 3
        g.mask_index = mask_index
        sl_labels = torch.argmax(g.x, dim=-1)
        g.label = sl_labels
        # pre-replace
        new_ids = np.random.choice(all_node_ids, size=len(replace_ids), replace=True)
        replace_labels = g.label[replace_ids].numpy()
        new_labels = g.label[new_ids].numpy()
        is_equal = (replace_labels == new_labels)
        while np.sum(is_equal):
            new_ids[is_equal] = np.random.choice(all_node_ids, size=np.sum(is_equal), replace=True)
            new_labels = g.label[new_ids].numpy()
            is_equal = (replace_labels == new_labels)
        g.x[replace_ids] = g.x[new_ids].clone()


class CollatorIPAPretrain(object):
    def __init__(self, candi_rate=0.7, mask_rate=0.8, replace_rate=0.1, keep_rate=0.1):
        self.candi_rate = candi_rate
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.keep_rate = keep_rate

    def __call__(self, g_lst):
        max_length = max([g.x.size(0) for g in g_lst])
        num_res = g_lst[0].atom_pos.size(1)
        x_dim = g_lst[0].x.size(1)
        batch_x = torch.zeros((len(g_lst), max_length, x_dim))
        batch_x_pos = torch.zeros((len(g_lst), max_length, num_res, 3))

        batch_x_pad = torch.zeros((len(g_lst), max_length), dtype=torch.bool)
        batch_x_mask = torch.zeros((len(g_lst), max_length), dtype=torch.long)
        # convert batch_label elements to 21
        batch_label = torch.zeros((len(g_lst), max_length), dtype=torch.long) + 20

        for i, g in enumerate(g_lst):
            length = g.x.size(0)
            sl_labels = torch.argmax(g.x, dim=-1)
            batch_x[i, :length] = g.x
            batch_x_pos[i, :length] = g.atom_pos
            batch_x_pad[i, :length] = True
            batch_label[i, :length] = sl_labels

            node_ids = np.arange(length, dtype=np.int64)
            candi_ids = np.random.choice(node_ids, size=int(length * self.candi_rate), replace=False)
            mask_ids = np.random.choice(candi_ids, size=int(len(candi_ids) * self.mask_rate), replace=False)

            candi_ids = np.setdiff1d(candi_ids, mask_ids)
            replace_ids = np.random.choice(candi_ids, size=int(
                len(candi_ids) * (self.replace_rate / (self.replace_rate + self.keep_rate))), replace=False)
            keep_ids = np.setdiff1d(candi_ids, replace_ids)

            # pre-replace
            curr_labels = sl_labels[replace_ids].numpy()
            new_labels = np.random.randint(0, 20, size=len(replace_ids))
            is_equal = (curr_labels == new_labels)
            while np.sum(is_equal):
                new_labels[is_equal] = np.random.randint(0, 20, size=np.sum(is_equal))
                is_equal = (curr_labels == new_labels)
            # transform new_labels into one-hot
            new_replace_x = F.one_hot(torch.tensor(new_labels), num_classes=20)
            new_replace_x = new_replace_x.float()
            batch_x[i, replace_ids] = new_replace_x

            batch_x_mask[i, mask_ids] = 1
            batch_x_mask[i, replace_ids] = 2
            batch_x_mask[i, keep_ids] = 3

        return batch_x, batch_x_pos, batch_x_pad, batch_x_mask, batch_label


class CollatorDiff(object):
    def __init__(self):
        pass

    def __call__(self, g_lst):
        g_batch = Batch.from_data_list(g_lst)
        max_length = max([g.x.size(0) for g in g_lst])
        num_res = g_lst[0].atom_pos.size(1)
        x_dim = g_lst[0].x.size(1)
        batch_x = torch.zeros((len(g_lst), max_length, x_dim))
        batch_x_pos = torch.zeros((len(g_lst), max_length, num_res, 3))
        batch_x_pad = torch.zeros((len(g_lst), max_length), dtype=torch.bool)

        batch_x_mask = torch.zeros((len(g_lst), max_length), dtype=torch.long)
        batch_label = torch.zeros((len(g_lst), max_length), dtype=torch.long) + 20

        for i, g in enumerate(g_lst):
            length = g.x.size(0)
            sl_labels = torch.argmax(g.x, dim=-1)
            batch_x[i, :length] = g.x
            batch_x_pos[i, :length] = g.atom_pos
            batch_x_pad[i, :length] = True
            batch_label[i, :length] = sl_labels
        ipa_batch = Data(x=batch_x, atom_pos=batch_x_pos, x_pad=batch_x_pad, x_mask=batch_x_mask, label=batch_label)
        return g_batch, ipa_batch
