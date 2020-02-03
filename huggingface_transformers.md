be# Use the model with huggingface/transformers

FinBERT is included in transformers, so you can simply use the model as usual:

```
model = transformers.BertForMaskedLM.from_pretrained("bert-base-finnish-cased-v1") 
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-finnish-cased-v1")
```
Use "bert-base-finnish-uncased-v1" if you want to use the uncased model.


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
mkdir bert-base-finnish-cased-transformers-v1
python3 -m transformers.convert_bert_original_tf_checkpoint_to_pytorch --tf_check bert-base-finnish-cased/bert-base-finnish-cased-v1 --bert_config bert-base-finnish-cased-v1/bert_config.json --pytorch bert-base-finnish-cased-transformers-v1/pytorch_model.bin
cp ./bert-base-finnish-cased-v1/bert_config.json bert-base-finnish-cased-transformers-v1/config.json
cp ./bert-base-finnish-cased-v1/vocab.txt bert-base-finnish-cased-transformers-v1/
```
