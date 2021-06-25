import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class Loss_S_Probe(nn.Module):
    def __init__(self, ) -> None:
        super(Loss_S_Probe, self).__init__()

    def forward(self, input: Tensor, input_mask: Tensor, target: Tensor):
        '''
        :param input:  predict probe (B,max_seq_len,max_seq_len)
        :param target: parsed tree probe (B,max_seq_len,max_seq_len)
        :return: loss scalar
        '''
        loss = 0.
        B, max_len, _ = input.size()
        if B == 0:
            return loss

        for i in range(B):
            raw_sent_len = input_mask[i].sum() - 2
            if raw_sent_len<=0: # 有些sample的句子是空的，会导致出现nan错误，因此需要判断是否句子为空
                loss+=0
                continue
            cur_loss = torch.abs(target[i, :raw_sent_len, :raw_sent_len] - input[i, 1:1 + raw_sent_len,
                                                                           1:1 + raw_sent_len] ** 2).mean()  # 删除CLS 和 SEP token的结果
            loss += cur_loss
        return loss / B
        # assert input.size() == target.size()
        # B, max_seq_len, _ = input.size()
        # loss = torch.abs(input - target).mean()
        # return loss


class Loss_SCL(nn.Module):
    '''
    this loss function is ported from:
    https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py#L76
    '''

    def __init__(self, temperature=0.07, n_views=2) -> None:
        super(Loss_SCL, self).__init__()
        self.n_views = n_views  # 'Number of views for contrastive learning training. default=2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, vis_probe_vec: Tensor, sent_probe_vec: Tensor):
        B, dim = vis_probe_vec.size()
        assert B % self.n_views == 0
        self.batch_size = B // self.n_views

        vis_logits, vis_labels = self.info_nce_loss(vis_probe_vec)
        vis_loss = self.criterion(vis_logits, vis_labels)

        sent_logits, sent_labels = self.info_nce_loss(sent_probe_vec)
        sent_loss = self.criterion(sent_logits, sent_labels)
        return vis_loss + sent_loss

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for _ in range(self.n_views)],
                           dim=0)  # [0,1,2,3, for n_views times] if batch size =4
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.long).bool().to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels


class Loss_XCL(nn.Module):
    def __init__(self, temperature=0.07, n_views=2) -> None:
        super(Loss_XCL, self).__init__()
        self.n_views = n_views  # 'Number of views for contrastive learning training. default=2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, vis_probe_vec: Tensor, sent_probe_vec: Tensor):
        B, dim = vis_probe_vec.size()
        assert B % self.n_views == 0
        self.batch_size = B // self.n_views

        vis_logits, vis_labels = self.info_nce_loss(vis_probe_vec, sent_probe_vec)
        vis_loss = self.criterion(vis_logits, vis_labels)

        sent_logits, sent_labels = self.info_nce_loss(sent_probe_vec, vis_probe_vec)
        sent_loss = self.criterion(sent_logits, sent_labels)
        return vis_loss + sent_loss

    def info_nce_loss(self, query_features, ref_features):
        labels = torch.cat([torch.arange(self.batch_size) for _ in range(self.n_views)],
                           dim=0)  # [0,1,2,3, for n_views times] if batch size =4
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        query_features = F.normalize(query_features, dim=1)
        ref_features = F.normalize(ref_features, dim=1)

        similarity_matrix = torch.matmul(query_features, ref_features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.long).bool().to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels




