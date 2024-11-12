retrieval_model_name=tapas_dual_encoder_proj_256_large
gsutil cp "gs://tapas_models/2021_04_27/${retrieval_model_name}.zip" . && unzip "${retrieval_model_name}.zip"