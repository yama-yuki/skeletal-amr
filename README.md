# Dependency Matching System: Complex Sentence to Skeletal AMR
Code for the dependency matching system presented in our \*SEM 2021 paper.

Citation:
- Yuki Yamamoto, Yuji Matsumoto, and Taro Watanabe. Dependency Patterns of Complex Sentences and Semantic Disambiguation for Abstract Meaning Representation Parsing. In Proceedings of the \*SEM 2021: The Tenth Joint Conference on Lexical and Computational Semantics.
```bib
@inproceedings{yamamoto-etal-2021-skel-amr,
  title     = {Dependency Patterns of Complex Sentences and Semantic Disambiguation for Abstract Meaning Representation Parsing},
  author    = {Yamamoto, Yuki and Matsumoto, Yuji and Watanabe, Taro},
  booktitle = {Proceedings of *SEM},
  year      = {2021},
  url       = {},
  pages     = {}
}
```

# Description

Macro and micro F1 scores of our classification models:

| Vanilla Softmax | *F<sub>M</sub>* | *F<sub>m</sub>* | Restricted Softmax | *F<sub>M</sub>* | *F<sub>m</sub>* |ep,lr,batch|
|:---|:---:|:---:|:---|:---:|:---:|---:|
|BERT→AMR |64.06 |74.29 |BERT→AMR+*r* |67.11 |77.18 |10,5e-05,16|
|BERT→WIKI |47.67 |61.72 |\-|\-|\-|3,2e-05,64|
|BERT→MIX<sub>8k</sub> |67.12 |77.50 |BERT→MIX<sub>8k</sub>+*r* |70.76 |80.52 |10,5e-05,16|
|**BERT→WIKI→AMR** |**72.43** |**81.22** |**BERT→WIKI→AMR+*r*** |**75.65** |**83.94** |10,3e-05,32|

All results are achieved using 5-fold cross validation on `AMR` data.

# Usage
## The code has been tested on ...
- python 3.7.7
- pytorch 1.6.0
- spacy 2.3.2
- stanza 1.0.1

All dependencies are listed in requirements.txt.

## Setup

Via conda:
```sh
# Clone repository
git clone https://github.com/yama-yuki/skeletal-amr.git
# Create conda environment
conda create -n skel python=3.7
# Activate conda environment
conda activate skel
# Install all dependencies
pip install -r requirements.txt
```

Download models trained on full data:

[BERT→WIKI→AMR]()

## Run Dependency Matching System
```sh
$ python main.py
```

## For Reproduction
Create Dataset:
`AMR`, `WIKI`, or `MIX`
```sh
$ python repro/create_data.py -d [data]
```

Train your own classifier:
- Training
```sh
$ python repro/train.py -d [data] -e [epochs] -r [learning_rate] -b [batch_size]
```
- Evaluation
```sh
$ python repro/test.py -m [model]
```
