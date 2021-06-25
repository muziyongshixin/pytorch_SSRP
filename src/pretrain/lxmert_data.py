# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
import json
import random

import numpy as np
from torch.utils.data import Dataset

from param import args
from pretrain.qa_answer_table import AnswerTable
from utils import load_obj_tsv
from IPython import embed
import pickle
import os

TINY_IMG_NUM = 100
FAST_IMG_NUM = 5000

import socket

host_name = socket.gethostname()
if host_name == 'xxxxx':
    Split2ImgFeatPath = {
        'mscoco_train': '/ssd1/liyz/lxmert/data/mscoco_imgfeat/train2014_obj36.tsv',
        'mscoco_minival': '/ssd1/liyz/lxmert/data/mscoco_imgfeat/val2014_obj36.tsv',
        'mscoco_nominival': '/ssd1/liyz/lxmert/data/mscoco_imgfeat/val2014_obj36.tsv',
        'vgnococo': '/ssd1/liyz/lxmert/data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
    }
    print('run on {}\n using data path:{}'.format(host_name, str(Split2ImgFeatPath)))
else:
    Split2ImgFeatPath = {
        'mscoco_train': 'data/mscoco_imgfeat/train2014_obj36.tsv',
        'mscoco_minival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
        'mscoco_nominival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
        'vgnococo': 'data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
        'mscoco_train_aug': 'data/mscoco_imgfeat/train2014_obj36.tsv',
        'mscoco_minival_aug': 'data/mscoco_imgfeat/val2014_obj36.tsv',
        'mscoco_nominival_aug': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    }

Split2SentProbePath = {
    'mscoco_train_aug': 'data/probe/mscoco_train_prob_matrix.pickle',
    'mscoco_minival_aug': 'data/probe/mscoco_minival_prob_matrix.pickle',
    'mscoco_nominival_aug': 'data/probe/mscoco_nominival_prob_matrix.pickle'
}


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label


class LXMERTDataset:
    def __init__(self, splits: str, qa_sets=None):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """
        self.name = splits
        self.sources = splits.split(',')

        # Loading datasets to data
        self.data = []
        for source in self.sources:
            self.data.extend(json.load(open("data/lxmert/%s.json" % source)))
        print("Load %d data from %s" % (len(self.data), self.name))

        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(qa_sets)
        print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        # Modify the answers
        for datum in self.data:
            labelf = datum['labelf']
            for cat, labels in labelf.items():
                for label in labels:
                    for ans in list(label.keys()):
                        new_ans = self.answer_table.convert_ans(ans)
                        if self.answer_table.used(new_ans):
                            if ans != new_ans:
                                label[new_ans] = label.pop(ans)
                        else:
                            label.pop(ans)

    def __len__(self):
        return len(self.data)


def make_uid(img_id, dset, sent_idx):
    return "{}_{}_{}".format(img_id, dset, sent_idx)


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""


class LXMERTTorchDataset(Dataset):
    def __init__(self, dataset: LXMERTDataset, topk=-1, use_augmentation=False):
        super().__init__()
        self.raw_dataset = dataset
        self.task_matched = args.task_matched

        self.use_augmentation=use_augmentation
        if self.use_augmentation:
            used_sent_cat= ('mscoco','mscoco_rephrase')# default to use ('mscoco','mscoco_rephrase'),no use vqa data
        else:
            used_sent_cat=('mscoco')
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM

        # Load the dataset
        img_data = []
        for source in self.raw_dataset.sources:
            img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk))

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        used_data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                used_data.append(datum)

        # Flatten the dataset (into one sent + one image entries)

        ###====================== one datum sample is below =====================================
        # {'img_id': 'COCO_val2014_000000203564',
        #  'labelf': {'vqa': [{'10:10': 1}, {'no': 1, 'yes': 0.3}, {'clock': 1}]},
        #  'sentf': {'mscoco': ['A bicycle replica with a clock as the front wheel.',
        #                       'The bike has a clock as a tire.',
        #                       'A black metal bicycle with a clock inside the front wheel.',
        #                       'A bicycle figurine in which the front wheel is replaced with a clock\n',
        #                       'A clock with the appearance of the wheel of a bicycle '],
        #            'vqa': ['What is the clock saying the time is?',
        #                    'Is it possible to ride the bicycle?',
        #                    'What is the front wheel of the bike?']}}
        ##=======================================================================================
        self.data = []
        for datum in used_data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if used_sent_cat is not None  and sents_cat not in used_sent_cat:
                    continue # only use the specified sentence category ， default is ('mscoco','mscoco_rephrase')
                if sents_cat in datum['labelf']:
                    labels = datum['labelf'][sents_cat]
                else:
                    labels = None
                for sent_idx, sent in enumerate(sents):
                    if isinstance(sent,list): # this block code is for rephrase data
                        for j,s in enumerate(sent):
                            new_datum={
                                'uid': make_uid(datum['img_id'], sents_cat, '{}_{}'.format(sent_idx,j)),
                                'img_id': datum['img_id'],
                                'sent': s
                            }
                            if labels is not None:
                                new_datum['label'] = labels[sent_idx]
                            self.data.append(new_datum)
                    else:
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent
                        }
                        if labels is not None:
                            new_datum['label'] = labels[sent_idx]
                        self.data.append(new_datum)
        print("Use %d data in torch dataset" % (len(self.data)))  # self.data里包含vqa的文本数据

    def __len__(self):
        return len(self.data)


    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data) - 1)]
        img_id = datum['img_id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, 35)]
        return feat

    def get_img_feat(self,img_id, use_augmentation=False):

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        obj_labels = img_info['objects_id'].copy()
        obj_confs = img_info['objects_conf'].copy()
        attr_labels = img_info['attrs_id'].copy()
        attr_confs = img_info['attrs_conf'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()

        if use_augmentation: # 当前只有图片级别的水平翻转是支持的
            if random.randint(0,100)&1==1:
                boxes[:,(0,2)]= img_w-boxes[:,(2,0)] #注意左右翻转之后，左上角的横坐标变成了右上角横坐标
            else:
                boxes=boxes


        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)


        return (feats, boxes), (obj_labels, obj_confs), (attr_labels, attr_confs)


    def __getitem__(self, item: int):
        datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']


        # get img feats
        (feats, boxes), (obj_labels, obj_confs), (attr_labels, attr_confs)=self.get_img_feat(img_id,use_augmentation=self.use_augmentation)

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['sent']
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data) - 1)]
                while other_datum['img_id'] == img_id:
                    other_datum = self.data[random.randint(0, len(self.data) - 1)]
                sent = other_datum['sent']

        # Label, convert answer to id
        if 'label' in datum:
            label = datum['label'].copy()
            for ans in list(label.keys()):
                label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            label = None

        # Create target
        example = InputExample(
            uid, sent, (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label
        )
        return example


class LXMERTEvaluator:
    def __init__(self, dataset: LXMERTDataset):
        self.raw_dataset = dataset

        # Create QA Eval Data
        self.data = []
        for datum in self.raw_dataset.data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:  # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:  # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplemented


class SSRPTorchDataset(Dataset):
    def __init__(self, dataset: LXMERTDataset, topk=-1, img_feats_dir='data/img_feats', use_augmentation=True):
        super().__init__()
        self.raw_dataset = dataset
        self.task_matched = args.task_matched
        self.used_sent_category = {'mscoco', 'mscoco_rephrase'}
        self.img_feats_dir = img_feats_dir

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM

        # Load the dataset
        img_data = []
        for source in self.raw_dataset.sources:
            img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk))

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        self.img_augmentation_methods = ['img_hflip', 'roi_hflip', 'roi_r90', 'roi_r180', 'roi_r270',
                                         'roi_jit0.8', 'roi_jit1.2']
        self.use_data_augmentation = use_augmentation

        # Filter out the dataset
        used_data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                used_data.append(datum)

        # Flatten the dataset (into one sent + one image entries)

        ###====================== one datum sample is below =====================================
        # {'img_id': 'COCO_val2014_000000561629',
        #  'labelf': {'vqa': [{'carpet': 0.3, 'paper': 0.9, 'scissors': 0.9},
        #                     {'10': 0.6,
        #                      '16': 0.3,
        #                      '20': 0.3,
        #                      '44': 0.3,
        #                      '70': 0.3,
        #                      '72': 0.3,
        #                      '8': 0.3,
        #                      'lot': 0.3},
        #                     {'red': 0.3}]},
        #  'sentf': {'mscoco': ['A little boy with scissors and construction paper.',
        #                       'A toddler holds scissors up in the air around a big '
        #                       'mess of cut up paper.',
        #                       'A boy is cutting up pieces of construction paper.',
        #                       'A boy is sitting on a floor cutting up paper.',
        #                       'A small child is playing on the floor and is surrounded '
        #                       'by torn up pieces of paper.'],
        #            'vqa': ['What is this kid playing with?',
        #                    'How many pieces of paper are there?',
        #                    'What color is the paper on the floor?'],
        #             'mscoco_rephrase': [['A little boy with scissors and construction paper.',
        #                            'A boy with scissors and building paper.'],
        #                           ['A toddler holds a pair of scissors in the air around a large '
        #                            'jumble of sliced paper.',
        #                            'A child holds scissors in the air around a large pile of '
        #                            'shredded paper.'],
        #                           ['A boy cuts up building paper.',
        #                            'A boy cuts pieces of construction paper.'],
        #                           ['A boy sits on the floor and cuts paper.',
        #                            'The boy is sitting on the floor, cutting paper.'],
        #                           ['A small child plays on the floor and is surrounded by torn '
        #                            'pieces of paper.',
        #                            'A small child plays on the floor and is surrounded by torn '
        #                'pieces of paper.']]}}
        ##=======================================================================================
        self.data = []
        self.rephrase_data = {}
        for datum in used_data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat != 'mscoco':  # 只使用mscoco的caption数据
                    continue
                labels = None
                for sent_idx, sent in enumerate(sents):
                    if sent_idx >= 5:  # 每个imgid最多使用前5句caption，因为backtranslation得到的结果只有前5句话的结果
                        break

                    sent_id = make_uid(datum['img_id'], sents_cat, sent_idx)
                    new_datum = {
                        'uid': sent_id,
                        'img_id': datum['img_id'],
                        'sent': sent
                    }
                    if labels is not None:
                        new_datum['label'] = labels[sent_idx]
                    self.data.append(new_datum)

                    ## sentence data augmentation
                    if self.use_data_augmentation:
                        rephrased_sents = sentf['mscoco_rephrase'][sent_idx]
                        aug_datas = []
                        for j in range(min(len(rephrased_sents),2)): # some case have more than 2 rephrase sents
                            new_datum = {
                                'uid': make_uid(img_id=datum['img_id'], dset='mscoco_rephrase',
                                                sent_idx='{}_{}'.format(sent_idx, j)),
                                'img_id': datum['img_id'],
                                'sent': rephrased_sents[j]
                            }
                            aug_datas.append(new_datum)
                        self.rephrase_data[sent_id] = aug_datas
        print("Use %d data in torch dataset" % (len(self.data)))  # self.data里包含vqa的文本数据


        self.probe_matrix={}
        for source in self.raw_dataset.sources:
            cur_source_matrix_data=pickle.load(open(Split2SentProbePath[source],'rb'))
            self.probe_matrix.update(cur_source_matrix_data)






    def load_img_feats(self, feats_path, key='img_raw', **kwargs):
        # img_all_data = pickle.load(open(feats_path, 'rb'))
        # # Get image info
        # img_info = img_all_data[key]
        # obj_num = img_info['num_boxes']
        # feats = img_info['features'].copy()
        # boxes = img_info['boxes'].copy()
        # assert obj_num == len(boxes) == len(feats)
        # if 'only_feats' in kwargs:
        #     obj_labels = None
        #     obj_confs = None
        #     attr_labels = None
        #     attr_confs = None
        # else:
        #     obj_labels = img_info['objects_id'].copy()
        #     obj_confs = img_info['objects_conf'].copy()
        #     attr_labels = img_info['attrs_id'].copy()
        #     attr_confs = img_info['attrs_conf'].copy()
        #
        # # Normalize the boxes (to 0 ~ 1)
        # img_h, img_w = img_info['img_h'], img_info['img_w']
        # boxes = boxes.copy()
        # boxes[:, (0, 2)] /= img_w
        # boxes[:, (1, 3)] /= img_h
        # np.testing.assert_array_less(boxes, 1 + 1e-5)
        # np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # Get image info
        img_id=feats_path.split('/')[-1].split('.')[0]
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        obj_labels = img_info['objects_id'].copy()
        obj_confs = img_info['objects_conf'].copy()
        attr_labels = img_info['attrs_id'].copy()
        attr_confs = img_info['attrs_conf'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        return (feats, boxes), (obj_labels, obj_confs), (attr_labels, attr_confs)

    def load_sent_probe(self,img_id, uid): # todo  to implement

        # sent_probe = np.zeros((36,36))
        sent_probe = self.probe_matrix[img_id][uid]
        return sent_probe

    def __getitem__(self, item: int):
        datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['sent']
        sent_probe = self.load_sent_probe(img_id,uid)

        feats_path = os.path.join(self.img_feats_dir, '{}.pickle'.format(img_id))
        (feats, boxes), (obj_labels, obj_confs), (attr_labels, attr_confs) = self.load_img_feats(feats_path)

        label = None
        # Create target
        example = InputExample(
            uid, (sent, sent_probe), (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label
        )
        if not self.use_data_augmentation:
            return example
        else:  # get augmentation data

            rephrased_sents = self.rephrase_data[uid]
            chosen_sent = random.choice(rephrased_sents)
            r_sent = chosen_sent['sent']
            r_uid = chosen_sent['uid']
            r_sent_probe = self.load_sent_probe(img_id,r_uid)

            img_aug_method = random.choice(self.img_augmentation_methods)
            (r_feats, r_boxes), (r_obj_labels, r_obj_confs), (r_attr_labels, r_attr_confs) = self.load_img_feats(
                feats_path, key=img_aug_method)

            r_example = InputExample(
                r_uid, (r_sent, r_sent_probe), (r_feats, r_boxes),
                (r_obj_labels, r_obj_confs), (r_attr_labels, r_attr_confs),
                is_matched, label
            )

            return example, r_example

    def __len__(self):
        return len(self.data)

