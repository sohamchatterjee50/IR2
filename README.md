# **[Re] On the Reproducibility of Open Domain Question Answering over Tables via Dense Retrieval**

This is the repository for the reproduction of the **Open Domain Question Answering over Tables via Dense Retrieval** paper. Many parts of the code are taken/adapted directly from the original repository [here](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md).

## **Setup**

For all the following steps, we highly suggest first entering the repository folder by doing:

```sh
cd ir2
```

First, we need the environment. To set it up, please run the following:

```sh
conda env create -f environment.yml
```

Also, before we continue, do note that you can adjust the configurations for each script listed below in their respective config files. They can be found [here](configs) and are named based on which file uses them.

**TIP:** To create new configs without editing the default ones, you can create a copy, change the name and desired values, and use them in the following way when running the `.py` scripts below:

```sh
python "[SCRIPT_NAME].py" --config-name="[YOUR_CUSTOM_CONFIG_IN_CONFIGS].yaml"
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

It is possible to generate the data manually via the following script:
```sh
python nq_preprocess.py
```

However, it is not recommended as it will take a long time.


### **Model Checkpoints**

The model checkpoint can be downloaded via the following command:

```sh
retrieval_model_name=tapas_nq_hn_retriever_medium
gsutil cp "gs://tapas_models/2021_04_27/${retrieval_model_name}.zip" . && unzip "${retrieval_model_name}.zip"
rmdir $retrieval_model_name

reader_model_name=tapas_nq_reader_large
gsutil cp "gs://tapas_models/2021_04_27/${reader_model_name}.zip" . && unzip "${reader_model_name}.zip"
rmdir $reader_model_name

# Alternatively, you can run these bash scripts:
bash/download_retriever.bash
bash/download_reader.bash
```

You can change the retrieval model name to any one of the checkpoints listed [here](misc/model_list.md). Currently, the above lines install the versions used in the original paper.

## **Running**

### **Retrieval**

There are several steps present to run all the experiments, which are outlined below:

1.  Model pre-training.
2.  Model fine-tuning.
3.  The selection of the best checkpoint with respect to a retrieval metric (i.e., `eval_recall_at_1`) in the local setting (which considers all tables that appear in the dev set as the corpus). These metrics are printed to XM.
4.  Producing global predictions for the selected best checkpoint - these consist of representations for all tables in the corpus.
5.  Generating retrieval metrics based on the global setting, and write KNN tables ID's and scores for each query to a JSON file (to be used for negatives mining or End2End QA).


We must thus first create the data to use. This can be done with the following script (note that running the below script is very time-consuming and resource intensive due to the sheer data size):
```sh
python convert_data.py
```

Then we can run the dual encoder experiment:
```sh
python retrieval_main.py
```

By default, the above script only performs prediction, as training the model fully is outside the scope of this study.

Now, all that needs to be done is to run the evaluation to print recall@k scores in the global setting given the best
model (e.g., 5K checkpoint in this case). This process generates the KNN of the most similar tables per query and their similarity scores to a `jsonl` file.

*   Set `prediction_files_local` to the best model output. This file holds the query ids, their representations, and the ids for the gold table.
*   Set `prediction_files_global` to the output path of the last step.

```sh
# Set the step parameter value according to the best dev results. The train and tables predictions generated in the previous step will only exist for this step.
python eval_retriever.py
```

### **Reading**

We now move on to the reading phase. 
First, we create the training data:
```sh
python create_e2e.py
python reader_main.py 
```

Now we can run the main experiment script, which can be achieved by changing the mode to `predict_and_evaluate` in the `reader_main.yaml`:
```sh
python reader_main.py 
```

This should output an `events.out.tfevents.` file, which can be opened with the following:
```sh
tensorboard --logdir="[FOLDER_CONTAINING_TFEVENTS]"
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
