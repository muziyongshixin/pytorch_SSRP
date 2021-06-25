import stanza
import json
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Pool, cpu_count
from queue import Queue
import time
from multiprocessing import Manager
import multiprocessing


use_multi_process=True
multi_num=4


cur_nlp_processor = stanza.Pipeline('en', "./", use_gpu=True)

def prepare_inputs(json_path):
    json_data = json.load(open(json_path))
    all_data = []
    for ele in tqdm(json_data):
        raw = ele['sentf']['mscoco']
        reph = ele['sentf']['mscoco_rephrase']
        cur_sample_sents = raw + [y for x in reph for y in x]
        all_data.append((ele['img_id'], cur_sample_sents))
    print('prepare finished, data path={}  all data num={}'.format(json_path, len(all_data)))
    return all_data


def process_one(ele_data, return_dict):
    img_id,cur_sample_sents=ele_data
    print(img_id, flush=True)
    in_docs = [stanza.Document([], text=d) for d in cur_sample_sents]  # Wrap each document with a stanza.Document object
    # cur_nlp_processor = nlp_queue.get(block=True)
    result = cur_nlp_processor(in_docs)
    assert len(cur_sample_sents) == len(result)

    cur_sample_result = []
    for i in range(len(result)):
        doc = result[i]
        if len(doc.sentences) > 0:
            sent = doc.sentences[0]
            cur = [{'id': word.id, 'word': word.text, 'head_id': word.head, 'head': sent.words[
                word.head - 1].text if word.head > 0 else "root", 'deprel': word.deprel}
                   for word in sent.words]
        else:
            print('something wrong with doc')
            pprint(img_id)
            cur = []
        cur_sample_result.append(cur)

    return_dict[img_id]=cur_sample_result



def process_all(json_path,save_path):

    start_time=time.time()
    thread_pool = Pool(multi_num)
    print('using {} process...'.format(multi_num))

    all_data=prepare_inputs(json_path)

    manager=Manager()
    return_dict = manager.dict()

    result=[]
    for ele in all_data[:]:
        # cur_nlp_processor = nlp_queue.get(block=True)
        result.append(thread_pool.apply_async(func=process_one, args=(ele, return_dict)))
        # nlp_queue.put(cur_nlp_processor)

    thread_pool.close()
    thread_pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    merged_result={}
    for key,val in return_dict.items():
        merged_result[key]=val

    json.dump(merged_result,open(save_path,'w'))
    print('finished all process in {} s, save {} samples to {}'.format(time.time()-start_time,len(return_dict),save_path))


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    all_paths=[('../lxmert/mscoco_train_aug.json','../probe/mscoco_train_prob.json'),
               ('../lxmert/mscoco_minival_aug.json','../probe/mscoco_minival_prob.json'),
               ('../lxmert/mscoco_nominival_aug.json','../probe/mscoco_nominival_prob.json')]

    for json_path,save_path in all_paths:
        process_all(json_path,save_path)


#
# nlp = stanza.Pipeline('en',  "./",use_gpu=True)  # Build the pipeline, specify part-of-speech processor's batch size
#
# minval_data = json.load(open('../lxmert/mscoco_minval_aug.json'))
# nominval_data = json.load(open('../lxmert/mscoco_nominval_aug.json'))
# train_data = json.load(open('../lxmert/mscoco_train_aug.json'))
#
#
# minval_data_result={}
# for ele in tqdm(minval_data):
#     raw = ele['sentf']['mscoco']
#     reph = ele['sentf']['mscoco_rephrase']
#     cur_sample_sents = raw + [y for x in reph for y in x]
#     # cur_sample_sents=['hello my name is liyongzhi, I am a student in china']*100
#     in_docs = [stanza.Document([], text=d) for d in
#                cur_sample_sents]  # Wrap each document with a stanza.Document object
#     result = nlp(in_docs)
#     assert len(cur_sample_sents) == len(result)
#
#     cur_sample_result = []
#     for i in range(len(result)):
#         doc = result[i]
#         if len(doc.sentences)>0:
#             sent = doc.sentences[0]
#             cur = [{'id': word.id, 'word': word.text, 'head_id': word.head, 'head': sent.words[
#                 word.head - 1].text if word.head > 0 else "root", 'deprel': word.deprel}
#                    for word in sent.words]
#         else:
#             print('something wrong with doc')
#             pprint(ele)
#             cur=[]
#         cur_sample_result.append(cur)
#     minval_data_result[ele['img_id']]=cur_sample_result
#
# json.dump(minval_data_result,open('../lxmert/mscoco_minval_prob.json','w'))
# print('finished minval data processing...')
#
