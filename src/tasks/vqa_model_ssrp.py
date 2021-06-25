# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from param import args
from ssrp.entry_ssrp import SSRP
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 36


class VQAModel_SSRP(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build ssrp encoder
        self.ssrp_encoder = SSRP(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        feats_dim = self.ssrp_encoder.feats_dim
        probe_dim = self.ssrp_encoder.probe_dim
        hid_dim = 768

        self.probe_feats_trans = nn.Sequential(
            nn.Linear(in_features=probe_dim * 2, out_features=probe_dim),
            GeLU(),
            BertLayerNorm(probe_dim, eps=1e-12),
            nn.Linear(probe_dim, hid_dim)
        )

        self.g_align = nn.Sequential(
            nn.Linear(in_features=feats_dim * 2, out_features=feats_dim),
            GeLU(),
            BertLayerNorm(feats_dim, eps=1e-12),
            nn.Linear(feats_dim, hid_dim)
        )

        self.fq = nn.Sequential(
            nn.Linear(in_features=hid_dim * 2, out_features=hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, hid_dim * 2)
        )

        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.probe_feats_trans.apply(self.ssrp_encoder.encoder.init_bert_weights)
        self.g_align.apply(self.ssrp_encoder.encoder.init_bert_weights)
        self.fq.apply(self.ssrp_encoder.encoder.init_bert_weights)
        self.logit_fc.apply(self.ssrp_encoder.encoder.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        # x = self.lxrt_encoder(sent, (feat, pos))
        # logit = self.logit_fc(x)

        (lang_output, visn_output), pooled_output, (
        vis_probe, lang_probe, vis_probe_vec, lang_probe_vec) = self.ssrp_encoder(sent, feat,
                                                                                  visual_attention_mask=None, pos=pos)

        B, _, _ = vis_probe.size()

        v_ = visn_output.mean(dim=1)
        f_align = self.g_align(torch.cat([v_, pooled_output], dim=-1))

        f_probe = self.probe_feats_trans(torch.cat([vis_probe.view(B, -1), lang_probe.view(B, -1)], dim=-1))

        q = self.fq(torch.cat((f_align, f_probe), dim=-1))
        logit = self.logit_fc(q)
        return logit
