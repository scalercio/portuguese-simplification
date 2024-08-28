# portuguese-simplification
## Installing
### 1. Create a fresh environment

I will be using conda but you can use your choice of environment management tool.

```
conda create -n acl24 python==3.10
```

### 2. Install PyTorch 2.1 (nightly)


```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

This will download CUDA driver version 11.8 if it is compatible with your installed CUDA runtime version.

### 3. Install other libraries

Let's install other requirements

```
pip install -r requirements.txt tokenizers sentencepiece huggingface_hub wandb>=0.12.10
```


## How to use

### Mined data
Portuguese paraphrase pairs and contexts that were mined from the common crawl data are located in the data/ccnet/ folder. These data are necessary for the training.

### Train the model
After adjusting the hyperparameters in the file scripts/train.py, run the command. 
```python
python scripts/train.py 
```
Please disregard the variable features_kwargs in the train.py file.

### Evaluate a trained model
After setting the variables model_version and model_paths in the file scripts/eval_models.py, run the command below. The default test set is porsimplessent. To change this behaviour, one must change the source/utils.py file.
```python
python scripts/eval_models.py 
```


## Authors

* **Arthur Scalercio** ([arthurscalercio@gmail.com](mailto:arthurscalercio@gmail.com))
* **Maria José Finatto** ([mariafinatto@gmail.com](mailto:mariafinatto@gmail.com))
* **Aline Paes** ([alinepaes@ic.uff.br](mailto:alinepaes@ic.uff.br))


## Citation

If you use this study in your research, please cite [Enhancing Sentence Simplification in Portuguese: Leveraging Paraphrases, Context, and Linguistic Features](https://aclanthology.org/2024.findings-acl.895.pdf)

```
@inproceedings{scalercio-etal-2024-enhancing,
    title = "Enhancing Sentence Simplification in {P}ortuguese: Leveraging Paraphrases, Context, and Linguistic Features",
    author = "Scalercio, Arthur  and
      Finatto, Maria  and
      Paes, Aline",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.895",
    pages = "15076--15091",
    abstract = "Automatic text simplification focuses on transforming texts into a more comprehensible version without sacrificing their precision. However, automatic methods usually require (paired) datasets that can be rather scarce in languages other than English. This paper presents a new approach to automatic sentence simplification that leverages paraphrases, context, and linguistic attributes to overcome the absence of paired texts in Portuguese.We frame the simplification problem as a textual style transfer task and learn a style representation using the sentences around the target sentence in the document and its linguistic attributes. Moreover, unlike most unsupervised approaches that require style-labeled training data, we fine-tune strong pre-trained models using sentence-level paraphrases instead of annotated data. Our experiments show that our model achieves remarkable results, surpassing the current state-of-the-art (BART+ACCESS) while competitively matching a Large Language Model.",
}
```