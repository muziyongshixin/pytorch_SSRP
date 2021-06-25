import numpy as np
from tqdm import tqdm
import json
import networkx as nx
import pickle
import time


def make_uid(img_id, dset, sent_idx):
    return "{}_{}_{}".format(img_id, dset, sent_idx)


def get_one_sent_probe(sent):
    G = nx.Graph()
    all_edge = []
    for edge in sent:
        u = edge['id']
        v = edge['head_id']
        all_edge.append((u, v))
    G.add_edges_from(all_edge)
    #     print(G.number_of_edges())
    gen = nx.all_pairs_shortest_path(G)
    shortest_path = dict(gen)
    probe_size = len(sent)
    probe = np.ones((probe_size, probe_size)) * -1
    for i in range(probe_size):
        for j in range(probe_size):
            probe[i][j] = len(shortest_path[i + 1][j + 1]) - 1  # stanza的结果单词从1开始编号
    return probe


def generate_probe_matrix(json_path, save_path):
    start_time = time.time()
    data = json.load(open(json_path))

    all_result = {}
    for img_id, img_sample in tqdm(data.items()):
        raw_sent_num = len(img_sample) - 10

        img_result = {}
        for i, sent in enumerate(img_sample):
            if i < raw_sent_num:
                sent_cat = 'mscoco'
                sent_idx = i
            else:
                sent_cat = 'mscoco_rephrase'
                raw_idx = (i - raw_sent_num) // 2
                j = (i - raw_sent_num) & 1
                sent_idx = '{}_{}'.format(raw_idx, j)

            key = make_uid(img_id, sent_cat, sent_idx)
            probe_matrix = get_one_sent_probe(sent)
            img_result[key] = probe_matrix
        all_result[img_id] = img_result

    pickle.dump(all_result, open(save_path, 'wb'))
    print('save probe matrix data to {}, total data number is {}, using time is {}'.format(save_path, len(all_result),
                                                                                           time.time() - start_time))



json_path='/m/liyz/lxmert/data/probe/mscoco_minival_prob.json'
save_path='/m/liyz/lxmert/data/probe/mscoco_minival_prob_matrix.pickle'

nominival_json_path='/m/liyz/lxmert/data/probe/mscoco_nominival_prob.json'
nominival_save_path='/m/liyz/lxmert/data/probe/mscoco_nominival_prob_matrix.pickle'
generate_probe_matrix(nominival_json_path,nominival_save_path)

trian_json_path='/m/liyz/lxmert/data/probe/mscoco_train_prob.json'
train_save_path='/m/liyz/lxmert/data/probe/mscoco_train_prob_matrix.pickle'
generate_probe_matrix(trian_json_path,train_save_path)