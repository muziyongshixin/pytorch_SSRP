# Self-Supervised Relationship Probing



## Introduction
This is an unofficial PyTorch implementation of NeurIPS2020 paper [Self-Supervised Relationship Probing"](https://proceedings.neurips.cc/paper/2020/hash/13f320e7b5ead1024ac95c3b208610db-Abstract.html).
Most codes in this project is ported from the repos below:
- https://github.com/sthalles/SimCLR (Contrastive learning part)

- https://github.com/john-hewitt/structural-probes (language probing part)

- https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md (Back-translation)

- https://github.com/MILVLG/bottom-up-attention.pytorch (VIsual Feature Extraction)

- https://github.com/airsplay/lxmert (Attention/BERT Part)

We thank the original author for his selfless help and answers during the implementation of this project.
 
 
Due to the limit time we only implement the `SSRP_Cross` model in the original paper, and test the model in the VQA2.0 task.
We train the encoder for 10 epochs in stage1 and finetune the encoder for 30 epochs in stage2 on VQA2.0 dataset.
The result is shown in below:
## Results on [VQA2.0](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) (with this Github version)
| Split(Test-Standard)           |  Binary| Number  | Other  | ACCu |
|-----------       |:----:   |:---:    |:------:|:------:|
| Paper Result | 87.8  | 54.4  | 62.7 | 72.0 |
| Our Implementation    | 86.4 | 49.0   | 58.9 | 69.11|

It should be clarified that we did not use the ROI augmentation strategy, and some hyper-parameters may different from the original paper.



## Data Preparation
See the description in the [VQA section](###VQA).


## Pretrain
### Stage1
1. Before run the model on whole training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `stage1_tiny` is the name of this experiment.
    ```bash
    bash stage1.bash 0 stage1_tiny --tiny
    ```
2. If no bug came out, then the model is ready to be trained on the whole VQA corpus:
    ```bash
    bash stage1.bash 0 stage1
    ```
### Stage2
1. Train the model on stage2 with the command below.
    ```bash
    bash stage2.bash 0 vqa_finetune_tiny --tiny --load_lxmert=$YOUR_STAGE1_CKPT.pth$
    ```
2. If no bug came out, then the model is ready to be trained on the whole training set:
    ```bash
    bash stage2.bash 0 vqa_finetune --load_lxmert=$YOUR_STAGE1_CKPT.pth$
 
   
## VQA
#### Fine-tuning
1. Please make sure the LXMERT pre-trained model is  [pre-trained](##Pretrain).

2. Download the re-distributed json files for VQA 2.0 dataset. The raw VQA 2.0 dataset could be downloaded from the [official website](https://visualqa.org/download.html).
    ```bash
    mkdir -p data/vqa
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/vqa/
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/vqa/
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/vqa/
    ```
3. Download faster-rcnn features for MS COCO train2014 (17 GB) and val2014 (8 GB) images (VQA 2.0 is collected on MS COCO dataset).
The image features are
also available on Google Drive and Baidu Drive (see [Alternative Download](#alternative-dataset-and-features-download-links) for details).
    ```bash
    mkdir -p data/mscoco_imgfeat
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
    ```

4. Before fine-tuning on whole VQA 2.0 training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `vqa_lxr955_tiny` is the name of this experiment.
    ```bash
    bash stage2.bash 0 vqa_finetune_tiny --tiny
    ```
5. If no bug came out, then the model is ready to be trained on the whole VQA corpus:
    ```bash
    bash stage2.bash 0 vqa_finetune
    ```

#### Local Validation
The results on the validation set (our minival set) are printed while training.
The validation result is also saved to `snap/vqa/[experiment-name]/log.log`.
If the log file was accidentally deleted, the validation result in training is also reproducible from the model snapshot:
```bash
bash run/vqa_test_ssrp.bash 0 vqa_ssrp_results --test minival --load=$YOUR_CKPT_PATH$
```
#### Submitted to VQA test server
1. Download our re-distributed json file containing VQA 2.0 test data.
    ```bash
    wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/test.json -P data/vqa/
    ```
2. Download the faster rcnn features for MS COCO test2015 split (16 GB).
    ```bash
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/test2015_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/test2015_obj36.zip -d data && rm data/mscoco_imgfeat/test2015_obj36.zip
    ```
3. Since VQA submission system requires submitting whole test data, we need to run inference over all test splits 
(i.e., test dev, test standard, test challenge, and test held-out). 
It takes around 10~15 mins to run test inference (448K instances to run).
    ```bash
    bash run/vqa_test_ssrp.bash 0 vqa_ssrp_results --test test --load=$YOUR_CKPT_PATH$
    ```
 The test results will be saved in `snap/vqa_ssrp_results/test_predict.json`. 
The VQA 2.0 challenge for this year is host on [EvalAI](https://evalai.cloudcv.org/) at [https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview)
It still allows submission after the challenge ended.
Please check the official website of [VQA Challenge](https://visualqa.org/challenge.html) for detailed information and 
follow the instructions in [EvalAI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to submit.
In general, after registration, the only thing remaining is to upload the `test_predict.json` file and wait for the result back.


### GQA & Other Tasks
More detailed experiments instructions can be found in the [LXMERT repo](https://github.com/airsplay/lxmert)
