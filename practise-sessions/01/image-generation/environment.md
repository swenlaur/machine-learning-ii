We need only keras and jupyter. But the environment was a bit excessive
```
conda create -n huggingface python=3.10
conda activate huggingface
conda install conda-forge::huggingface_hub
conda install jupyter
conda install conda-forge::tf-keras
conda install anaconda::diffusers-torch
pip install "diffusers>=0.29.0"
conda install conda-forge::matplotlib
pip install -q git+https://github.com/tensorflow/docs
conda install conda-forge::imageio
```
