# Learning Visual Representation from Modality-Shared Contrastive Language-Image Pre-training (MS-CLIP)

This repo contains the source code of our ECCV 2022 paper MS-CLIP:

[**Learning Visual Representation from Modality-Shared Contrastive Language-Image Pre-training**](https://openreview.net/pdf?id=ROteIE-4A6W)
<br>
2022 European Conference on Computer Vision (ECCV 2022)
<br>
By Haoxuan You*, Luowei Zhou*, Bin Xiao*, Noel Codella*, Yu Cheng, Ruochen Xu, Shih-Fu Chang, Lu Yuan.


## Introduction
![MS-CLIP](/Figs/diagram1.png)

we investigate a variety of Modality-Shared Contrastive Language-Image Pre-training (MS-CLIP) frameworks. More specifically, we question how many parameters of a transformer model can be shared across modalities during contrastive pre-training, and rigorously examine architectural design choices that position the proportion of parameters shared along a spectrum. In studied conditions, we observe that a mostly unified encoder for vision and language signals outperforms all other variations that separate more parameters. Additionally, we find that light-weight modality-specific parallel modules further improve performance.

![MS-CLIP-S](/Figs/diagram2.png)


## Update
- [07/20/2022] Released pretrained model and zero-shot evaluation on ImageNet-1k.

## Pre-trained Weights
| Model | Training Set | Top-1 on IN-1K | LP on 24 datasets | Download
| :----: | :---: | :---: | :---: | :---: |
| MS-CLIP-S (ViT-B/32) | YFCC-22M | 36.7 | 68.5 | [ckpt](https://1drv.ms/u/s!ApxSG5cLDhMkg4M4h8GPb7QYsUoM6Q)/[config](experiments/model/b32-yfcc-msclips.yaml)
| MS-CLIP-S (ViT-B/16) | YFCC-22M | 39.0 | 70.4 | [ckpt](https://1drv.ms/u/s!ApxSG5cLDhMkg4M3PYoDKdcrR7NOgQ?e=ExKwEW)/[config](experiments/model/b16-yfcc-msclips.yaml)
| MS-CLIP-S (ViT-B/32) |LAION-20M| 40.2 | 73.3 | [ckpt](https://1drv.ms/u/s!ApxSG5cLDhMkg4M57QFPJzPrMReF8A?e=eGW8NX)/[config](experiments/model/b32-laion-msclips.yaml)



## Getting Started
### Installation
Please follow [INSTALL.md](./INSTALL.md) for installation
### Data preparation
Please following [DATA.md](./DATASET/DATA.md) for data preparation.
### Pre-trained weights preparation
Download from the links in the table above. Put the weights under `./OUTPUT_MODEL/`.
### Evaluation
To evaluate a pre-trained MS-CLIP on ImageNet Zero-shot Classification, run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_zeroshot.py --model <config-file> 
```
where `<config-file>` is the config yaml under `experiments/model/`. E.g. `experiments/model/b32-laion-msclips.yaml`


<!-- ## Citation
If you find this project useful for your research, please kindly cite our paper:

```bibtex
@incollection{NIPS2019_8940,
title = {PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation},
author = {Qin, Can and You, Haoxuan and Wang, Lichen and Kuo, C.-C. Jay and Fu, Yun},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {7190--7201},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8940-pointdan-a-multi-scale-3d-domain-adaption-network-for-point-cloud-representation.pdf}
}
``` -->

## Contact
If you have any questions, please contact [Haoxuan You](haoxuanyou@gmail.com) or [Luowei Zhou](zhouluoweiwest@gmail.com).
