How to train your ~dragon~ BERT
===============================

Once the tf_records have been generated, we move on to training with that data.

The code to call is [`run_pretraining.py`](https://github.com/haamis/DeepLearningExamples_FinBERT/blob/master/TensorFlow/LanguageModeling/BERT_nonscaling/run_pretraining.py). Example of how to run with sbatch (this is on CSC's Puhti):

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
# Limit on Puhti is 3 days, we'll have to run this multiple times.
#SBATCH -t 72:00:00
#SBATCH -J finnish_5_9_final_data
#SBATCH -o /scratch/project_2001553/rami/horovod_logs/finnish_5_9_final_data_out-%j.txt
#SBATCH -e /scratch/project_2001553/rami/horovod_logs/finnish_5_9_final_data_err-%j.txt
# Allocate 4 GPUs on each node.
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4
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
# Batch size is the max amount we can fit into VRAM, seq_length is 128 for the first part of the training.
# Max_predictions_per_seq is the default and must be the same as set in the tfrecord generation.
# Lastly, `--horovod` enables Horovod support, `--xla` enables TF's XLA JIT and `--use_fp16` enables support for mixed-precision training.
srun python run_pretraining.py --input_file=/scratch/project_2001553/data-sep-2019/finnish/tfrecords/cased/128/* --output_dir=$OUTPUT_DIR --do_train=True --do_eval=False --bert_config_file=$BERT_DIR/finnish_main_config_50k.json --train_batch_size=140 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=900000 --num_warmup_steps=9000 --learning_rate=1e-4 --horovod --use_xla --use_fp16
seff $SLURM_JOBID
```
