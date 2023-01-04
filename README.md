# Practical and Efficient Model Extraction of Sentiment Analysis APIs
Pytorch implementation for the ICSE 2023 paper. In this repository, we release the code for stealing the BERT model trained on the MR dataset. We will release the complete code if this paper gets accepted. 

## Additional results
Tests of Statistical Signifcance

- P-values of t-tests when comparing our method with the state-of-the-art baseline in terms of the design of the proxy training source

![plot](./img/proxy.png)

- P-values of t-tests when comparing our method with different sampling strategies

![plot](./img/sampling.png)



## Requirements

- Python 3.6.13
- Cuda 10.2
- Pytorch 1.10.1
- textattack 0.3.0

## Experiments

You should install the textattack environment from (https://github.com/QData/TextAttack) before running the code, which includes the pretrained weights and other required packages.

### Introduction

- `wiki_data_sub.tsv` : the randomly selected data from the wiki-103 dataset as the query samples.

- `bert_mr_predict_wiki.py` : the implementation to query the victim model and generate labeled dataset .

- `policy.py` : the implementation of our proposed method to train the extracted model.

- `evaluation.py` : the code for evaluating the extracted model.

  

### Example Usage

#### Generate labeled dataset by the victim model:

- BERT-MR

```
python bert_mr_predict_wiki.py
```

#### Train the extracted model

- XLNet

```
python policy.py
```

#### Evaluate the extracted model

```
python evaluation.py
```
