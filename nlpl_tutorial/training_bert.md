# How to train your ~dragon~ BERT

Once the tf_records have been generated, we move on to training with that data.

We follow the training schedule used in the original BERT: training for 1M steps using Adam with a learning rate of 1e-4 and a warmup of 1% followed by linear decay.
Training is done in two phases: with sequence lengths 128 and 512. 900k steps are done with length 128 and 100k steps with 512 length. This is done for computational efficiency, as the attention mechanism scales quadratically with sequence length and the position embeddings for sequence length 512 can be learned fairly quickly in the last part of training.

We used [Nvidia's modified version](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) of Google's original code, with minor modifications. Note that Nvidia has since updated their repository with some changes that may not work with our sbatch files, but I adapting is *probably* quite easy. Our modified version is [here](https://github.com/haamis/DeepLearningExamples_FinBERT/tree/master/TensorFlow/LanguageModeling/BERT_nonscaling), and links to files are in this document are to our fork.

The code to call is [`run_pretraining.py`](https://github.com/haamis/DeepLearningExamples_FinBERT/blob/master/TensorFlow/LanguageModeling/BERT_nonscaling/run_pretraining.py). [Example of an sbatch file](../nlpl_tutorial/finnish_5_9_final_data.sbatch) to run on CSC's Puhti:

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

This will then be called with `sbatch <file>`. The code will go through about 260k steps in 3 days, so multiple sequential jobs will be needed. The easiest way to do this (in my experience) is to queue them with `sbatch --dependency=singleton <file>`. This only runs one job with a given name from the same user at a time. More info in [SLURM's documentation](https://slurm.schedmd.com/sbatch.html).

Once the 900k steps of the 128 phase are done, the 512 phase is run with a [slightly modified sbatch file](../nlpl_tutorial/finnish_5_9_final_data_512.sbatch). The differences to the example above are:
  * `max_seq_length` set to 512
  * input files changed to tfrecords created with 512 seq_length
  * `max_predictions_per_seq` set to the same as when generating the tfrecords (around 77-80)
  * `batch_size` set lower, 20 fits in memory
  * `num_train_steps` set to 1000000
