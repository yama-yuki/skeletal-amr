# Dependency Matching System: Complex Sentence to Skeletal AMR
Code for the dependency matching system presented in our STARSEM2021 paper.

Citation:
- Yuki Yamamoto, Yuji Matsumoto, and Taro Watanabe. 2021. Dependency Patterns of Complex Sentences and Semantic Disambiguation for Abstract Meaning Representation Parsing. In Proceedings of the STARSEM.
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

> since

# Usage
## 0. Setup

Via conda:
```sh
git clone https://github.com/yama-yuki/skeletal-amr.git
conda create -n skel python=3.7
conda activate skel
pip install -r requirements.txt
```

## 1. Run Dependency Matching System
```sh
$ python main.py
```

## 2. Train Your Own Classifier
- Training
```sh
$ python train.py -d [data]
```
- Evaluation
```sh
$ python test.py
```

## The code has been tested on ...
- python 3.7.7
- pytorch 1.6.0
- spacy 2.3.2
- stanza 1.0.1
