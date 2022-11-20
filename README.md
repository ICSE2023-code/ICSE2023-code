# Practical and Efficient Model Extraction of Sentiment Analysis APIs
Pytorch implementation for ICSE 2023 paper "Practical and Efficient Model Extraction of Sentiment Analysis APIs". In this repository, we release the code for stealing BERT model trained on MR dataset. We will release other code if this paper is accepted. 

## Requirements

- Python 3.6.13
- Cuda 10.2
- Pytorch 1.10.1
- textattack 0.3.0

## Experiments

You should install the textattack environment from (https://github.com/QData/TextAttack) before running the code, which includes the pretrained weights and other required packages.

#### Introduction

- `wiki_data_sub.tsv` : the randomly selected data from the wiki-103 dataset as the query samples.

- `bert_mr_predict_wiki.py` : the implementation to query the victim model and generate labeled dataset .

- `policy.py` : the implementation of our proposed method to train the extracted model.

- `evaluation.py` : the code for evaluating the extracted model.

  

#### Example Usage

##### Generate labeled dataset by the victim model:

- BERT-MR

```
python bert_mr_predict_wiki.py
```

##### Train the extracted model

- XLNet

```
python policy.py
```

##### Evaluate the extracted model

```
python evaluation.py
```

#### Results

##### Agreement (%) between the extracted model and MR-BERT for different sampling strategies
![](https://github.com/ICSE2023-code/ICSE2023-code/blob/main/images/bert-mr.png)

##### Agreement (%) between the extracted model and MR-XLNet for different sampling strategies
![](https://github.com/ICSE2023-code/ICSE2023-code/blob/main/images/bert-mr.png)
