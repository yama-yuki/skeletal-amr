# Dependency Matching System: Complex Sentence to Skeletal AMR
Code for the dependency matching system presented in our \*SEM 2021 paper.

Citation:
- Yuki Yamamoto, Yuji Matsumoto, and Taro Watanabe. 2021. Dependency Patterns of Complex Sentences and Semantic Disambiguation for Abstract Meaning Representation Parsing. In Proceedings of the \*SEM 2021: The Tenth Joint Conference on Lexical and Computational Semantics.
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

| Models | *F<sub>M</sub>* | *F<sub>m</sub>* |
|:---|:---:|:---:|
|BERT→AMR |64.06 |74.29 |
|BERT→WIKI |47.67 |61.72 |
|BERT→MIX<sub>8k</sub> |67.12 |77.50 |
|BERT→WIKI→AMR |72.43 |81.22 |

# Usage
## Setup

Via conda:
```sh
git clone https://github.com/yama-yuki/skeletal-amr.git
conda create -n skel python=3.7
conda activate skel
pip install -r requirements.txt
```

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

## The code has been tested on ...
- python 3.7.7
- pytorch 1.6.0
- spacy 2.3.2
- stanza 1.0.1
