# portuguese-simplification
### 1. Create a fresh environment

I will be using conda but you can use your choice of environment management tool.

```
conda create -n acl24 python==3.10
```

### 2. Install PyTorch 2.1 (nightly)


```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

This will download CUDA driver version 11.8 if it is compatible with your installed CUDA runtime version. We will revisit CUDA again.

### 3. Install other libraries

Let's install other requirements by lit-gpt

```
pip install -r requirements.txt tokenizers sentencepiece huggingface_hub wandb>=0.12.10
```