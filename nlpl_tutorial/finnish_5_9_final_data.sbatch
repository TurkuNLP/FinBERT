#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH -J finnish_5_9_final_data
#SBATCH -o /scratch/project_2001553/rami/horovod_logs/finnish_5_9_final_data_out-%j.txt
#SBATCH -e /scratch/project_2001553/rami/horovod_logs/finnish_5_9_final_data_err-%j.txt
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4
#SBATCH --account=Project_2001553
#SBATCH

umask 0007

module purge
module load tensorflow/1.13.1-hvd

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BERT_DIR=/users/ilorami1/DeepLearningExamples/TensorFlow/LanguageModeling/BERT_nonscaling/
export OUTPUT_DIR=/scratch/project_2001553/rami/pretraining/finnish_5_9_final_data/

mkdir -p $OUTPUT_DIR

cd $BERT_DIR

export NCCL_DEBUG=INFO
srun python run_pretraining.py --input_file=/scratch/project_2001553/data-sep-2019/finnish/tfrecords/cased/128/* --output_dir=$OUTPUT_DIR --do_train=True --do_eval=False --bert_config_file=$BERT_DIR/finnish_main_config_50k.json --train_batch_size=140 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=900000 --num_warmup_steps=9000 --save_checkpoints_steps=10000 --learning_rate=1e-4 --horovod --use_xla --use_fp16
seff $SLURM_JOBID
