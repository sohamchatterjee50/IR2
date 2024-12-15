step=50000
model_dir='/media/stefan/My Passport/NQ-Tables/nq-tables2/models/nq/'

python tapas/scripts/eval_table_retriever.py \
 --prediction_files_local=${model_dir}/train/predict_results_${step}.tsv \
 --prediction_files_global=${model_dir}/tables/predict_results_${step}.tsv \
 --retrieval_results_file_path="${model_dir}/train_knn.jsonl"

# Computes test results
python tapas/scripts/eval_table_retriever.py \
 --prediction_files_local=${model_dir}/test/predict_results_${step}.tsv \
 --prediction_files_global=${model_dir}/tables/predict_results_${steps}.tsv \
 --retrieval_results_file_path="${model_dir}/test_knn.jsonl"

# Computes dev results
python tapas/scripts/eval_table_retriever.py \
 --prediction_files_local=${model_dir}/eval_results_${step}.tsv \
 --prediction_files_global=${model_dir}/tables/predict_results_${steps}.tsv \
 --retrieval_results_file_path="${model_dir}/dev_knn.jsonl"
