
## Installation Instructions

- Clone this repo:

```bash
git clone https://github.com/Hxyou/MSCLIP
cd MSCLIP
```

- Create a conda virtual environment and activate it (optional):

```bash
conda create -n msclip python=3.7 -y
conda activate msclip
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.6.0` and `torchvision==0.7.0` with `CUDA==10.1`:

```bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```


- Install other requirements:

```bash
pip install numpy scikit-learn Pillow tqdm yacs json ftfy regex transformers einops pickle tensorwatch
```