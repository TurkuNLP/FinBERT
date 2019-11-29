## Release 1.0

**November 25, 2019**

Download the models here:

* Cased Finnish BERT Base: [bert-base-finnish-cased-v1.zip](http://dl.turkunlp.org/finbert/bert-base-finnish-cased-v1.zip)
* Uncased Finnish BERT Base: [bert-base-finnish-uncased-v1.zip](http://dl.turkunlp.org/finbert/bert-base-finnish-uncased-v1.zip)

We generally recommend the use of the cased model.

(These models are identical to previously released other than naming.)

### Usage

If you want to use the model with the huggingface/transformers library, follow the steps in [huggingface_transformers.md](https://github.com/TurkuNLP/FinBERT/blob/master/huggingface_transformers.md)

### Results

Initial, as of yet unpublished and therefore unofficial evaluation results of the model are as follows:

#### Named Entity Recognition

Evaluation on FiNER corpus ([Ruokolainen et al 2019](https://arxiv.org/abs/1908.04212))

| Model          | Accuracy |
|--------------------|----------|
| **FinBERT-Base Cased**  | **92.40%** |
| BERT-Base Multilingual Cased (Google) | 90.29% |
| Rule-based (FiNER) | 86.82%      |

[code](https://github.com/jouniluoma/keras-bert-ner), [data](https://github.com/mpsilfve/finer-data)

(FiNER tagger results from [Ruokolainen et al. 2019](https://arxiv.org/pdf/1908.04212.pdf))

#### PoS tagging

UD_Finnish-TDT test set, gold segmentation

| Model                         |      |
|-------------------------------|------|
| **FinBERT-Base Cased**          | **98.23%** |
| BERT-Base Multilingual Cased (Google) | 96.97% |

[code](https://github.com/spyysalo/bert-pos), [data](http://hdl.handle.net/11234/1-2837)

## Previous releases

### Release 0.2

**October 24, 2019** Beta version of the BERT base uncased model trained from scratch on a corpus of Finnish news, online discussions, and crawled data. 

Download the model here: [bert-base-finnish-uncased.zip](http://dl.turkunlp.org/finbert/bert-base-finnish-uncased.zip)

### Release 0.1

**September 30, 2019** We release a beta version of the BERT base cased model trained from scratch on a corpus of Finnish news, online discussions, and crawled data. 

Download the model here: [bert-base-finnish-cased.zip](http://dl.turkunlp.org/finbert/bert-base-finnish-cased.zip)
