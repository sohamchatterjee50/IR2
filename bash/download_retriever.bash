# For the retriever model
retrieval_model_name=tapas_nq_hn_retriever_medium
gsutil cp "gs://tapas_models/2021_04_27/${retrieval_model_name}.zip" . && unzip "${retrieval_model_name}.zip"
rmdir $retrieval_model_name