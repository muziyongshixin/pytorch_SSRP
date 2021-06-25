# coding=utf-8
# Copyleft 2019 project LXRT.
# import os
# import sys
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 找到当前项目的项目的路径
# print(base_dir)
# sys.path.append(base_dir)  # 将找到的项目的路径导入当前系统路径

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 找到当前项目的项目的路径
print(base_dir)
sys.path.append(base_dir)  # 将找到的项目的路径导入当前系统路径

import collections
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from param import args
from pretrain.lxmert_data import InputExample, LXMERTDataset, SSRPTorchDataset, LXMERTEvaluator
# from lxrt.entry import set_visual_config
from lxrt.tokenization import BertTokenizer
from lxrt.modeling import SSRP_Encoder, SSRP_Probe
from lxrt.modeling import VISUAL_CONFIG
from lxrt.loss_functions import Loss_S_Probe, Loss_XCL, Loss_SCL
import wandb

wandb.init(settings=wandb.Settings(start_method='fork'),
           project="ssrp_finetune_stage2",
           notes=args.remark,
           tags=["baseline", "stage2"],
           entity='muziyongshixin')


DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')


def my_collate_fn(x):
    assert len(x) > 0
    tuple_size = len(x)
    if len(x[0]) == 1:
        return x
    else:  # 说明使用了data augmentation
        raw_data = [t[0] for t in x]
        aug_data = [t[1] for t in x]
        return raw_data + aug_data


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1) -> DataTuple:
    # Decide which QA datasets would be used in pre-training.
    # Options: vqa, gqa, visual7w
    # Note: visual7w is a part of vgqa, we take the name here.
    qa_sets = args.qa_sets
    if qa_sets is not None:
        qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))

    # Build dataset, data loader, and evaluator.
    dset = LXMERTDataset(splits, qa_sets=qa_sets)
    tset = SSRPTorchDataset(dset, topk, img_feats_dir='data/img_feats', use_augmentation=True)

    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=my_collate_fn,
        drop_last=drop_last, pin_memory=True
    )
    evaluator = LXMERTEvaluator(dset)
    print('finished {} get_tuple process...'.format(splits))
    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)


valid_batch_size = (50 * torch.cuda.device_count()) if args.multiGPU else 64
valid_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, drop_last=False, topk=5000)

train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True)
print('finished all data preparation process ....')


class SSRPInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, lm_label_ids, sent_probe,
                 visual_feats, obj_labels,
                 is_matched, ans):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.sent_probe = sent_probe
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels

        self.is_matched = is_matched

        self.ans = ans


LOSSES_NAME = ('s_probe', 'scl', 'xcl')


class SSRP:
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        if self.tokenizer is None:
            from IPython import embed
            embed()

        # Build model
        print('begin building models...')
        set_visual_config(args)

        self.encoder = SSRP_Encoder.from_pretrained(
            "bert-base-uncased",
            mode='lxr'
        )
        self.probe_head = SSRP_Probe()
        print('model construction is finished...')

        if args.load_lxmert is None:
            raise ValueError('encoder parameter is required...')
        # Load lxmert would not load the answer head.
        self.load_encoder(args.load_lxmert)
        if args.load_probe_head is not None:
            self.load_probe_head(args.load_probe_head)
        else:
            print('train the probe head from scratch...')

        # GPU Options
        self.encoder = self.encoder.cuda()
        self.probe_head = self.probe_head.cuda()
        if args.multiGPU:
            self.encoder = nn.DataParallel(self.encoder)
            self.probe_head = nn.DataParallel(self.probe_head)

        # keep the encoder with evaluation mode and freeze the parameter
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requeires_grad = False

        # loss functions
        self.Loss_S_Probe = Loss_S_Probe()
        self.Loss_SCL = Loss_SCL()
        self.Loss_XCL = Loss_XCL()

        # Optimizer
        train_ld = train_tuple.loader
        from lxrt.optimization import BertAdam
        batch_per_epoch = len(train_ld)
        t_total = int(batch_per_epoch * args.epochs)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print("Warm up Iters: %d" % warmup_iters)
        self.optim = BertAdam(self.probe_head.parameters(), lr=args.lr, warmup=warmup_ratio, t_total=t_total)

        wandb.config.update(args)

    def convert_example_to_features(self, example: InputExample, max_seq_length) -> SSRPInputFeatures:
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        (sent, sent_probe) = example.sent
        tokens = self.tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        # concatenate lm labels and account for CLS, SEP, SEP
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 获得sent probe matrix， 所有的大小都调整到max_seq_len，max_seq_len
        pad_sent_probe = np.zeros((max_seq_length, max_seq_length))  # 注意这里的矩阵不包含CLS和SEP token
        if len(sent_probe) > max_seq_length - 2:
            sent_probe = sent_probe[:max_seq_length - 2, :max_seq_length - 2]  # 最多取前面的max_seq_len-2个单词

        pad_sent_probe[:len(sent_probe), :len(sent_probe)] = sent_probe  # 有用的元素填充在左上角

        # Mask & Segment Word
        lm_label_ids = [-1] * len(input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length

        feat, boxes = example.visual_feats
        obj_labels, obj_confs = example.obj_labels
        attr_labels, attr_confs = example.attr_labels

        # QA answer label
        if example.label is None or len(example.label) == 0 or example.is_matched != 1:
            # 1. No label 2. Label is pruned 3. unmatched visual + language pair
            ans = -1
        else:
            keys, values = zip(*example.label.items())
            if len(keys) == 1:
                ans = keys[0]
            else:
                value_sum = sum(values)
                prob = [value / value_sum for value in values]
                choice = np.random.multinomial(1, prob).argmax()
                ans = keys[choice]

        features = SSRPInputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            lm_label_ids=lm_label_ids,
            sent_probe=pad_sent_probe,
            visual_feats=(feat, boxes),
            obj_labels={
                'obj': (obj_labels, obj_confs),
                'attr': (attr_labels, attr_confs),
                'feat': (None, None),
            },
            is_matched=example.is_matched,
            ans=ans,
        )
        return features

    def forward(self, examples):
        train_features = [self.convert_example_to_features(example, self.max_seq_length)
                          for example in examples]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        sent_target_probe = torch.tensor([f.sent_probe for f in train_features], dtype=torch.float).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()

        """
        SSRP_Encoder.forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None,pos=None,
                visual_attention_mask=None):
        """
        (lang_output, visn_output), pooled_output = self.encoder(input_ids, segment_ids, input_mask, visual_feats=feats,
                                                                 pos=pos)

        vis_probe, lang_probe, vis_probe_vec, lang_probe_vec = self.probe_head(lang_output, visn_output)

        losses = {}
        loss_s_probe = self.Loss_S_Probe(lang_probe, input_mask, sent_target_probe)
        loss_scl = self.Loss_SCL(vis_probe_vec, lang_probe_vec)
        loss_xcl = self.Loss_XCL(vis_probe_vec, lang_probe_vec)

        loss_cl_all = loss_s_probe + loss_scl + loss_xcl
        losses['s_probe'] = loss_s_probe.detach()
        losses['scl'] = loss_scl.detach()
        losses['xcl'] = loss_xcl.detach()
        return loss_cl_all, losses

    def train_batch(self, optim, batch):
        optim.zero_grad()
        loss, losses = self.forward(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(self.probe_head.parameters(), 1.)
        optim.step()
        return loss.item(), losses

    def valid_batch(self, batch):
        with torch.no_grad():
            loss, losses = self.forward(batch)
        return loss.item(), losses

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        train_ld = train_tuple.loader
        # Train
        best_eval_loss = 9595.
        for epoch in range(args.epochs):
            # Train
            print('====== begin training {} epoch ====='.format(epoch))
            batch_per_epoch = len(train_ld)
            total_loss = 0.
            total_losses = {n: 0. for n in LOSSES_NAME}
            for batch in tqdm(train_ld, total=len(train_ld)):
                loss, losses = self.train_batch(self.optim, batch)
                total_loss += loss
                for loss_name, val in losses.items():
                    total_losses[loss_name] += val

            print("The training loss for Epoch %d is %0.4f" % (epoch, total_loss / batch_per_epoch))
            losses_str = "The losses are "


            train_log = {'train_{}'.format(name): total_losses[name] for name in total_losses}
            wandb.log(train_log)
            for name, loss in total_losses.items():
                losses_str += "%s: %0.4f " % (name, loss / batch_per_epoch)

            print(losses_str)

            # Eval
            print('====== begin evaluate {} epoch ====='.format(epoch))
            avg_eval_loss = self.evaluate_epoch(eval_tuple, iters=-1)

            # Save
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                self.save("BEST_EVAL_LOSS")
            if epoch % 2 == 0:
                self.save("Epoch%02d" % (epoch + 1))

    def evaluate_epoch(self, eval_tuple: DataTuple, iters: int = -1):
        self.probe_head.eval()
        eval_ld = eval_tuple.loader
        total_loss = 0.
        total_losses = {n: 0. for n in LOSSES_NAME}
        for i, batch in enumerate(eval_ld):
            loss, losses = self.valid_batch(batch)
            total_loss += loss
            for loss_name, val in losses.items():
                total_losses[loss_name] += val
            if i == iters:
                break

        eval_log = {'eval_{}'.format(name): total_losses[name] for name in total_losses}
        wandb.log(eval_log)
        print("The valid loss is %0.4f" % (total_loss / len(eval_ld)))
        losses_str = "The losses are "
        for name, loss in total_losses.items():
            losses_str += "%s: %0.4f " % (name, loss / len(eval_ld))
        print(losses_str)

        return total_loss / len(eval_ld)

    def save(self, name):
        torch.save(self.probe_head.state_dict(),
                   os.path.join(args.output, "%s_probe_head.pth" % name))

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
        state_dict = torch.load("%s_probe_head.pth" % path)

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


if __name__ == "__main__":
    ssrp = SSRP(max_seq_length=36)

    ssrp.train(train_tuple, valid_tuple)
