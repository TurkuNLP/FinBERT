# FinBERT

BERT model trained from scratch on Finnish.

# Release 0.1

**September 30, 2019** We release a beta version of the BERT base cased model trained from scratch on a corpus of Finnish news, online discussions, and crawled data. 

[Download the model here.](http://dl.turkunlp.org/finbert/)

Initial, as of yet unpublished and therefore unofficial evaluation results of the model are as follows:

### Named Entity Recognition on the FiNER data

| Model          | Accuracy |
|--------------------|----------|
| Rule-based (FiNER) | 87%      |
| BERT-Base Multilingual Cased (Google) | 88% |
| FinBERT-Base Cased  | 91% |


### PoS tagging

UD_Finnish-TDT test set, gold segmentation

| Model                         |      |
|-------------------------------|------|
| BERT-Base Multilingual Cased (Google) | 96.93% |
| FinBERT-Base Cased          | 98.45% |

