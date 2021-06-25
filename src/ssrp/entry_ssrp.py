# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import os
# import sys
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  #找到当前项目的项目的路径
# print(base_dir)
# sys.path.append(base_dir)   #将找到的项目的路径导入当前系统路径

import torch
import torch.nn as nn

from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature
from lxrt.modeling import VISUAL_CONFIG
from lxrt.modeling import SSRP_Encoder, SSRP_Probe

import os


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers


class SSRP(nn.Module):
    def __init__(self, args, max_seq_length, mode='lxr'):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(args)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        self.encoder = SSRP_Encoder.from_pretrained(
            "bert-base-uncased",
            mode=mode
        )
        self.probe_head = SSRP_Probe()
        print('model construction is finished...')

        if args.from_scratch:
            print("initializing all the weights")
            self.encoder.apply(self.encoder.init_bert_weights)
        elif args.load is None:
            if args.load_lxmert is None:
                raise ValueError('encoder parameter is required...')
            # Load lxmert would not load the answer head.
            self.load_encoder(args.load_lxmert)
            if args.load_probe_head is None:
                raise ValueError('probe head parameter is required...')
            else:
                self.load_probe_head(args.load_probe_head)


    def multi_gpu(self):
        self.encoder = nn.DataParallel(self.encoder)
        self.probe_head = nn.DataParallel(self.probe_head)

    @property
    def feats_dim(self):
        return 768

    @property
    def probe_dim(self):
        return 36*36

    def forward(self, sents, feats, visual_attention_mask=None,pos=None):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        (lang_output, visn_output), pooled_output = self.encoder(input_ids, segment_ids, input_mask, visual_feats=feats,
                                                                 pos=pos)

        vis_probe, lang_probe, vis_probe_vec, lang_probe_vec = self.probe_head(lang_output, visn_output)

        # output = self.model(input_ids, segment_ids, input_mask,
        #                     visual_feats=feats,
        #                     visual_attention_mask=visual_attention_mask)
        return (lang_output, visn_output), pooled_output, (vis_probe, lang_probe, vis_probe_vec, lang_probe_vec)


    def load_encoder(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load(path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.encoder.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.encoder.load_state_dict(state_dict, strict=False)

    def load_probe_head(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load( path)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.probe_head.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.probe_head.load_state_dict(state_dict, strict=False)
