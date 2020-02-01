# How to train your ~dragon~ BERT

## General info

We follow the training schedule used in the original [BERT paper](https://arxiv.org/abs/1810.04805): training for 1M steps using Adam with a learning rate of 1e-4 and a warmup of 1% followed by linear decay. The batch size used is 140\*8 GPUs = 1120. This is quite a lot more than the 256 used in the original, as we were attempting to make the training stable. Later experiments with BERT pretraining have indicated that using only 4 GPUs, and thus half the batch size, is still stable. Using 8 GPUs with the 512 sequence length is still recommended.

Training is done in two phases: with sequence lengths 128 and 512. 900k steps are done with length 128 and 100k steps with 512 length. This is done for computational efficiency, as the attention mechanism scales quadratically with sequence length and the position embeddings for sequence length 512 can be learned fairly quickly in the last part of training.

We use [Nvidia's modified version](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) of Google's original code, with minor modifications. The benefits in Nvidia's implementation are support for multi-GPU and multi-node training, [XLA](https://www.tensorflow.org/xla) support and half-precision support. The first one makes pretraining the model on GPUs feasible in the first place and the latter two bring impressive performance benefits. Note that Nvidia has since updated their repository with some changes that may not work with our sbatch files, but adapting is *probably* quite easy. Our modified version is [here](https://github.com/haamis/DeepLearningExamples_FinBERT/tree/master/TensorFlow/LanguageModeling/BERT_nonscaling), and links to files in this document are to our fork.

Our modifications are:
  * Removing learning rate scaling by number of GPUs added by Nvidia as this made the training unstable. [(link)](https://github.com/haamis/DeepLearningExamples_FinBERT/blob/master/TensorFlow/LanguageModeling/BERT_nonscaling/run_pretraining.py#L492)
  * Setting the code to save 100 most recent checkpoints. This means that when saving a checkpoint every 10000 steps, every checkpoint is kept from beginning to end. [(link)](https://github.com/haamis/DeepLearningExamples_FinBERT/blob/master/TensorFlow/LanguageModeling/BERT_nonscaling/run_pretraining.py#L481)

## Steps to take

First we must define a [`bert_config.json`](https://github.com/haamis/DeepLearningExamples_FinBERT/blob/master/TensorFlow/LanguageModeling/BERT_nonscaling/bert_config.json) file. The only thing that needs to be changed compared to the linked file is `"vocab_size"` which has to be equal to the size of your created vocabulary. This is just `wc -l <vocab file>`.

Next step is to run [`run_pretraining.py`](https://github.com/haamis/DeepLearningExamples_FinBERT/blob/master/TensorFlow/LanguageModeling/BERT_nonscaling/run_pretraining.py). [Example of an sbatch file](../nlpl_tutorial/finnish_5_9_final_data.sbatch) to run on CSC's Puhti:

```bash
#!/bin/bash

# Definining resource we want to allocate. We set 8 tasks, 4 tasks over 2 nodes as we have 4 GPUs per node.
#SBATCH --nodes=2
#SBATCH --ntasks=8

# 6 CPU cores per task to keep the parallel data feeding going. A little overkill, but CPU time is very cheap compared to GPU time.
#SBATCH --cpus-per-task=6

# Enough memory.
#SBATCH --mem=64G
#SBATCH -p gpu

# Time limit on Puhti's gpu partition is 3 days.
#SBATCH -t 72:00:00
#SBATCH -J finnish_5_9_final_data

# Log file locations, %j corresponds to slurm job id.
#SBATCH -o /scratch/project_2001553/rami/horovod_logs/finnish_5_9_final_data_out-%j.txt
#SBATCH -e /scratch/project_2001553/rami/horovod_logs/finnish_5_9_final_data_err-%j.txt

# Allocate 4 GPUs on each node.
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4

# Puhti project number, you'll have to change this.
#SBATCH --account=Project_2001553
#SBATCH

# This is used to make checkpoints and logs to readable and writable by other members in the project.
umask 0007

# Clear all modules and load Tensorflow with Horovod support.
module purge
module load tensorflow/1.13.1-hvd

# Some handy variables, you'll need to change these.
export BERT_DIR=/users/ilorami1/DeepLearningExamples/TensorFlow/LanguageModeling/BERT_nonscaling/
export OUTPUT_DIR=/scratch/project_2001553/rami/pretraining/finnish_5_9_final_data/

mkdir -p $OUTPUT_DIR

cd $BERT_DIR

export NCCL_DEBUG=INFO

# The actual command we want to run.
# Batch size is the max amount we can fit into VRAM, `--max_seq_length` is 128 for the first part of the training.
# `--max_predictions_per_seq` is the default and must be the same as set in the tfrecord generation.
# Lastly, `--horovod` enables Horovod support, `--xla` enables TF's XLA JIT and `--use_fp16` enables support for mixed-precision training.

srun python run_pretraining.py --input_file=/scratch/project_2001553/data-sep-2019/finnish/tfrecords/cased/128/* --output_dir=$OUTPUT_DIR --do_train=True --do_eval=False --bert_config_file=$BERT_DIR/finnish_main_config_50k.json --train_batch_size=140 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=900000 --num_warmup_steps=9000 --learning_rate=1e-4 --horovod --use_xla --use_fp16
seff $SLURM_JOBID
```

This will then be called with `sbatch <file>`. The code will go through about 260k steps in 3 days, so multiple sequential jobs will be needed. All the jobs could be queued at one time, however it is recommended to keep an eye on the training logs in case the training diverges, .

The easiest way to sequentially queue these runs (in my experience) is with `sbatch --dependency=singleton <file>`. This only runs one job with a given name from the same user at a time. More info in [SLURM's documentation](https://slurm.schedmd.com/sbatch.html).

Once the 900k steps of the 128 phase are done, the 512 phase is run with a [slightly modified sbatch file](../nlpl_tutorial/finnish_5_9_final_data_512.sbatch). The differences to the example above are:
  * `max_seq_length` set to 512.
  * Input files changed to tfrecords created with 512 seq_length.
  * `max_predictions_per_seq` set to the same as when generating the tfrecords (around 77-80).
  * `batch_size` set lower, 20 fits in memory.
  * `num_train_steps` set to 1000000.

## Uncased training
For training an uncased model you only need to change the input files, output directory and job name (for `--dependency=singleton` to work if training both models in parallel).

## Technical considerations on Puhti
When using XLA on Puhti you might come across errors complaining about a file called `libdevice` or `ptxas`. These errors are caused by Puhti's environment being slightly broken with regards to CUDA, possibly due to inflexibility on CUDA's side. The problem can be solved by creating a symlink to the files in the BERT directory. Currently the files are in `/appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/bin/ptxas` and `/appl/spack/install-tree/gcc-8.3.0/cuda-10.1.168-mrdepn/nvvm/libdevice/libdevice.10.bc`, however these locations may change with CUDA updates. The problem has been reported to CSC.
