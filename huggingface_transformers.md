# Use the model with huggingface/transformers

You need to tell the library where the model can be found like so (cut'n'paste to your code):

```
import transformers
transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/.torch-transformers/bert-base-finnish-cased/pytorch_model.bin"
transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/.torch-transformers/bert-base-finnish-cased/config.json"
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/.torch-transformers/bert-base-finnish-cased/vocab.txt"
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["bert-base-finnish-cased"]=512
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["bert-base-finnish-cased"]={'do_lower_case': False}
```

after which you can use the model as usual:

```
model = BertForMaskedLM.from_pretrained("bert-base-finnish-cased")
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
tokenizer = BertTokenizer.from_pretrained("bert-base-finnish-cased")
```

## Convert from tensorflow

This is mostly a "note-to-self" on how the model is converted from tensorflow checkpoint to the huggingface/transformers format.

```
python3 -m venv venv-transformers
source venv-transformers/bin/activate
pip3 install torch torchvision torchtext tensorflow-gpu
```

Then you can convert the model as follows:

**Note:** If you get the error *'BertForPreTraining' object has no attribute 'shape'* then you may need to edit `venv-transformers/lib64/python3.6/site-packages/transformers/modeling_bert.py` as instructed in [this issue](https://github.com/huggingface/transformers/issues/393).


```
mkdir bert-base-finnish-cased-transformers
python3 -m transformers.convert_bert_original_tf_checkpoint_to_pytorch --tf_check bert-base-finnish-cased/bert-base-finnish-cased --bert_config bert-base-finnish-cased/bert_config.json --pytorch bert-base-finnish-cased-transformers/pytorch_model.bin
cp ./bert-base-finnish-cased/bert_config.json bert-base-finnish-cased-transformers/config.json
cp ./bert-base-finnish-cased/vocab.txt bert-base-finnish-cased-transformers/
```
