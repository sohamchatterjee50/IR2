import os
import sys
import time 
sys.path.append('/gpfs/home5/scur2840/IR2/Authors_Code/tapas-master/tapas')
from tapas.utils import tf_example_utils
from tapas.utils import beam_runner
from tapas.utils import create_data
from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils
from tapas.scripts import prediction_utils
from tapas.scripts import eval_table_retriever_utils
from tapas.retrieval import tf_example_utils as retrieval_utils
from tapas.experiments import table_retriever_experiment
#import tensorflow as tf
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#nq_data_dir='/home/scur2840/IR2/Authors_Code/tapas-master/processed_table_data/nq-tables2/interactions/nq_tables_interactions_dev.tfrecord'
#retrieval_model_name='/home/scur2840/IR2/Authors_Code/tapas-master/tapas_dual_encoder_proj_256_tiny'
#model_dir='/home/scur2840/IR2/Authors_Code/tapas-master/processed_table_data/nq-tables2/models/nq'
#model_dir='/home/scur2840/IR2/Authors_Code/tapas-master/tapas_dual_encoder_proj_256_tiny'

# print("Is GPUs Available: ", tf.test.is_gpu_available())
# print("Is Cuda GPU available:",tf.test.is_gpu_available(cuda_only=True))
# os.system(f"python3   tapas/experiments/table_retriever_experiment.py \
#    --do_train \
#    --keep_checkpoint_max=40 \
#    --model_dir=\"{model_dir}\" \
#    --input_file_train=\"{nq_data_dir}/tf_examples/nq_tables_interactions_train.tfrecord\" \
#    --bert_config_file=\"{retrieval_model_name}/bert_config.json\" \
#    --init_checkpoint=\"{retrieval_model_name}/model.ckpt\" \
#    --init_from_single_encoder=false \
#    --down_projection_dim=256 \
#    --num_train_examples=5120000 \
#    --learning_rate=1.25e-5 \
#    --train_batch_size=256 \
#    --warmup_ratio=0.01 \
#    --max_seq_length=\"{max_seq_length}\"")


# os.system(f"python3   tapas/experiments/table_retriever_experiment.py \
#    --do_predict \
#    --model_dir=\"{model_dir}\" \
#    --input_file_eval=\"{nq_data_dir}/tf_examples/nq_tables_interactions_dev.tfrecord\" \
#    --bert_config_file=\"{retrieval_model_name}/bert_config.json\" \
#    --init_from_single_encoder=false \
#    --down_projection_dim=256 \
#    --eval_batch_size=32 \
#    --num_train_examples=5120000 \
#    --max_seq_length=\"{max_seq_length}\"")





max_seq_length=512
retrieval_model_name='/home/scur2840/IR2/Authors_Code/tapas-master/tapas_dual_encoder_proj_256_large'
nq_data_dir='/home/scur2840/IR2/Authors_Code/tapas-master/processed_table_data/nq-tables2'
#model_dir='/home/scur2840/checkpoints_v1'
model_dir=f"/scratch-shared/scur2840/checkpoints_50000/{time.strftime('checkpoint/%Y-%m-%d-%H-%M-%S')}" 
# os.system(f"python3 /home/scur2840/IR2/Authors_Code/tapas-master/tapas/experiments/table_retriever_experiment.py \
#    --do_train \
#    --keep_checkpoint_max=5 \
#    --model_dir=\"{model_dir}\" \
#    --input_file_train=\"{nq_data_dir}/tf_examples/nq_tables_interactions_train.tfrecord\" \
#    --bert_config_file=/home/scur2840/checkpoints/bert_config.json \
#    --init_checkpoint=/home/scur2840/checkpoints/model.ckpt-0 \
#    --init_from_single_encoder=false \
#    --down_projection_dim=256 \
#    --learning_rate=1.25e-5 \
#    --num_train_examples=51200 \
#    --save_checkpoints_steps=100 \
#    --train_batch_size=4 \
#    --warmup_ratio=0.01 \
#    --max_seq_length=\"{max_seq_length}\"")


os.system(f"python3  tapas/experiments/table_retriever_experiment.py \
     --do_predict \
     --evaluated_checkpoint_step=11300 --model_dir=/scratch-shared/scur2840/checkpoints_50000/checkpoint/2024-12-10-14-47-53 \
     --prediction_output_dir=\"{retrieval_model_name}/test_50000\" \
     --input_file_predict=\"{nq_data_dir}/tf_examples/nq_tables_interactions_test.tfrecord\" \
     --bert_config_file=/scratch-shared/scur2840/checkpoints_50000/checkpoint/2024-12-10-14-47-53/bert_config.json \
     --init_from_single_encoder=false \
     --down_projection_dim=256 \
     --eval_batch_size=32 \
     --max_seq_length=\"{max_seq_length}\"")

# os.system(f"python3   tapas/retrieval/create_retrieval_data_main.py \
#   --input_interactions_dir=\"{nq_data_dir}/sampled_interactions\" \
#   --input_tables_dir=\"{nq_data_dir}/tables\" \
#   --output_dir=\"{nq_data_dir}/tf_examples_sampled\" \
#   --vocab_file=\"{retrieval_model_name}/vocab.txt\" \
#   --max_seq_length=\"{max_seq_length}\" \
#   --max_column_id=\"{max_seq_length}\" \
#   --max_row_id=\"{max_seq_length}\" \
#   --use_document_title")