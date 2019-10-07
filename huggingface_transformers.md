# Use the model with huggingface/transformers

## Convert from tensorflow

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
