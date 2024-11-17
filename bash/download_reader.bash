# For the reader model
reader_model_name=tapas_nq_reader_larg
gsutil cp "gs://tapas_models/2021_04_27/${reader_model_name}.zip" . && unzip "${reader_model_name}.zip"
rmdir $reader_model_name