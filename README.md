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
Our dependency matching system is a pipeline of "Dependency Matcher" and "Relation Classifier". The pipeline first processes an input sentence with lexical and syntactic preprocessing using `SpaCy` and `Stanza`.

## Dependency Matcher

Our matcher is build upon [dependency matching module]() of `SpaCy`, which works in naive manner.
The dependency patterns are stored in `patterns`.

## Relation Classifier



Macro and micro F1 scores of our classification models:

| Vanilla Softmax | *F<sub>M</sub>* | *F<sub>m</sub>* | Restricted Softmax | *F<sub>M</sub>* | *F<sub>m</sub>* |ep, l_r, b_s|
|:---|:---:|:---:|:---|:---:|:---:|---:|
|BERT→AMR |64.06 |74.29 |BERT→AMR+*r* |67.11 |77.18 |10,5e-05,16|
|BERT→WIKI |47.67 |61.72 |\-|\-|\-|3,2e-05,64|
|BERT→MIX<sub>8k</sub> |67.12 |77.50 |BERT→MIX<sub>8k</sub>+*r* |70.76 |80.52 |10,5e-05,16|
|**BERT→WIKI→AMR** |**72.43** |**81.22** |**BERT→WIKI→AMR+*r*** |**75.65** |**83.94** |10,3e-05,32|

All results are achieved using 5-fold cross validation on `AMR` data.

All models are trained in the below environment:
```
OS: Ubuntu 16.04.7 LTS
GPU: GTX1080 Ti 
CPU: Xeon E5-2620 v4
```

# Usage
The code has been tested on ...
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

Download the best model trained on full data: 

- [BERT→WIKI→AMR]()

## Run Dependency Matching System
```sh
$ python main.py
```

## For Reproduction
1. Create Dataset:
```sh
$ python repro/create_data.py -d {data}
```
`-d`: choose data to create (`AMR`, `WIKI`, `MIX`)

Or use ours in `rsc`.

2. Train Classifier:
```sh
$ python repro/main.py -p {pretrained_model} -d {data} -e {epochs} -r {learning_rate} -b {batch_size}
```
`-p`: choose `bert-base-uncased` for pre-trained BERT or specify pre-finetuned model (e.g. `BERT-WIKI/3_2e-05_64`)

`-d`: data to finetune the model (`AMR`, `WIKI`, `MIX`)

`-e`: training epochs (`3`, `5`, `10`)

`-r`: initial learning rate (`2e-05`, `3e-05`, `5e-05`)

`-b`: batch size (`16`, `32`, `64`)

3. Evaluate:
```sh
$ python repro/main.py -t {trained_model}
```
`-t`: model to evaluate (e.g. `WIKI-AMR/WIKI_3_3e-05_64_AMR_10_3e-05_32`)
