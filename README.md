# **[Re] On the Reproducibility of Open Domain Question Answering over Tables via Dense Retrieval**

## **Setup**
To setup the code, ...

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
@inproceedings{herzig-etal-2020-tapas,
    title = "{T}a{P}as: Weakly Supervised Table Parsing via Pre-training",
    author = {Herzig, Jonathan  and
      Nowak, Pawel Krzysztof  and
      M{\"u}ller, Thomas  and
      Piccinno, Francesco  and
      Eisenschlos, Julian},
    editor = "Jurafsky, Dan  and
      Chai, Joyce  and
      Schluter, Natalie  and
      Tetreault, Joel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    doi = "10.18653/v1/2020.acl-main.398",
    pages = "4320--4333",
}
```
