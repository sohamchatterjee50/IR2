# **[Re] On the Reproducibility of Open Domain Question Answering over Tables via Dense Retrieval**

## **Setup**

For all the following steps, we highly suggest first entering the repository folder by doing:

```sh
cd ir2
```

First, we need the environment. To set it up, please run the following:

```sh
conda env create -f environment.yml
```

### **Dataset**

The dataset can be downloaded via the following command:

```sh
mkdir -p "data"
gsutil -m cp -R gs://tapas_models/2021_07_22/nq_tables/* "data"

# Alternatively, you can run this bash script:
bash/download_data.bash
```

Note that doing the above requires `gsutil` installed beforehand. Instructions on how to do so can be found [here](https://cloud.google.com/storage/docs/gsutil_install).


### **Model Checkpoints**

The model checkpoint can be downloaded via the following command:

```sh
retrieval_model_name=tapas_dual_encoder_proj_256_large
gsutil cp "gs://tapas_models/2021_04_27/${retrieval_model_name}.zip" . && unzip "${retrieval_model_name}.zip"

# Alternatively, you can run this bash script:
bash/download_ckpt.bash
```

You can change the retrieval model name to any one of the checkpoints listed [here](misc/model_list.md).

## **Running**

To run the code, use the following script:

```sh
python tapas/run_task_main.py \
  --task="SQA" \
  --output_dir="${output_dir}" \
  --init_checkpoint="${tapas_data_dir}/model.ckpt" \
  --bert_config_file="${tapas_data_dir}/bert_config.json" \
  --mode="predict_and_evaluate"
```

## **Citations**

To cite this work, please use the following citation:
```
@misc{...}
```

We recommend citing the original paper here:
```
@inproceedings{herzig-etal-2021-open,
    title = "Open Domain Question Answering over Tables via Dense Retrieval",
    author = {Herzig, Jonathan  and
      M{\"u}ller, Thomas  and
      Krichene, Syrine  and
      Eisenschlos, Julian},
    editor = "Toutanova, Kristina  and
      Rumshisky, Anna  and
      Zettlemoyer, Luke  and
      Hakkani-Tur, Dilek  and
      Beltagy, Iz  and
      Bethard, Steven  and
      Cotterell, Ryan  and
      Chakraborty, Tanmoy  and
      Zhou, Yichao",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.43",
    doi = "10.18653/v1/2021.naacl-main.43",
    pages = "512--519",
}

```

The license of the original code is also included [here](LICENSE) as this work is based on it.
