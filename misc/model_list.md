# **List of Model Checkpoint Names**

These are taken from the original repository:

## **Retrieval Models**

Size  |  Type           | Hard Negatives | Down Project | Recall@1 | Recall@10 | Recall@50 | Link
----- | --------------- | -------------- | ------------ | -------- | --------- | --------- | ----
LARGE | Pretrained      | No             | No           | | | | [  dual_encoder_proj_0_large.zip](https://storage.googleapis.com/  models/2021_04_27/  dual_encoder_proj_0_large.zip)
LARGE | Pretrained      | No             | 256          | | | | [  dual_encoder_proj_256_large.zip](https://storage.googleapis.com/  models/2021_04_27/  dual_encoder_proj_256_large.zip)
MEDIUM | Pretrained      | No             | 256          | | | | [  dual_encoder_proj_256_medium.zip](https://storage.googleapis.com/  models/2021_04_27/  dual_encoder_proj_256_medium.zip)
SMALL | Pretrained      | No             | 256          | | | | [  dual_encoder_proj_256_small.zip](https://storage.googleapis.com/  models/2021_04_27/  dual_encoder_proj_256_small.zip)
TINY  | Pretrained      | No             | 256          | | | | [  dual_encoder_proj_256_tiny.zip](https://storage.googleapis.com/  models/2021_04_27/  dual_encoder_proj_256_tiny.zip)
LARGE | Finetuned on NQ | No             | 256          | 35.9 | 75.9 | 91.4 | [  nq_retriever_large.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_retriever_large.zip)
LARGE | Finetuned on NQ | Yes            | 256          | 44.2 | 81.8 | 92.3 | [  nq_hn_retriever_large.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_hn_retriever_large.zip)
MEDIUM | Finetuned on NQ | No             | 256          | 37.1 | 74.5 | 88.0 | [  nq_retriever_medium.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_retriever_medium.zip)
MEDIUM| Finetuned on NQ | Yes            | 256          | 44.9 | 79.8 | 91.1 | [  nq_hn_retriever_medium.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_hn_retriever_medium.zip)
SMALL | Finetuned on NQ | No             | 256          | 37.6 | 72.8 | 87.4 | [  nq_retriever_small.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_retriever_small.zip)
SMALL | Finetuned on NQ | Yes            | 256          | 41.8 | 77.1 | 89.9 | [  nq_hn_retriever_small.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_hn_retriever_small.zip)
TINY | Finetuned on NQ | No             | 256          | 17.3 | 54.1 | 76.3 | [  nq_retriever_tiny.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_retriever_tiny.zip)
TINY | Finetuned on NQ | Yes            | 256          | 22.2 | 61.3 | 78.9 | [  nq_hn_retriever_tiny.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_hn_retriever_tiny.zip)


## **Reader Models**

Size  | Hard Negatives | Link
----- | -------------- | --------------
LARGE | No             | [  nq_reader_large.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_reader_large.zip)
LARGE | Yes            | [  nq_hn_reader_large.zip](https://storage.googleapis.com/  models/2021_04_27/  nq_hn_reader_large.zip)