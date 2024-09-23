# @BENCH: Are Vision-Language Models Ready for Assistive Technology?
\[[Project](https://junweizheng93.github.io/publications/ATBench/ATBench.html)\]\[[Paper](https://arxiv.org/pdf/2212.11270.pdf)\]

by Xin Jiang*, [Junwei Zheng*](https://junweizheng93.github.io/), [Ruiping Liu*](https://scholar.google.com/citations?user=tJYUHDgAAAAJ&hl=zh-CN), [Jiahang Li](https://www.researchgate.net/profile/Jiahang-Li), [Jiaming Zhang&dagger](https://jamycheung.github.io/), [Sven Matthiesen](https://scholar.google.com/citations?user=75P3ny0AAAAJ&hl=de), [Rainer Stiefelhagen](https://scholar.google.com/citations?user=SFCOJxMAAAAJ&hl=en) in WACV 2025

\* denotes equal contribution and &dagger denotes corresponding author 

## News

* **[2024.09.17]** ATBench is accepted to WACV2025.
* **[2024.09.17]** We are excited to release the @model training code ([INSTALL.md](asset/INSTALL.md), [DATASET.md](asset/DATASET.md), [TRAIN.md](asset/TRAIN.md), [EVALUATION.md](asset/EVALUATION.md))

<!-- <p align="center">
  <img src="/images/pipeline.png" width="90%" height="90%">
</p> -->
![pipeline](/images/pipeline.png)

## Introduction

![multi_task_result](/images/multi_task_result.png)

@Bench is designed by a pre-design user study with PVIs, including five five most crucial vision-language tasks: **Panoptic Segmentation**, **Image Captioning**, **Visual Question Answering (VQA)**, **Depth Estimation**, **Optical Character Recognition (OCR)**. And we also proposed a novel
@Model that can address all tasks simultaneously.

More detailed can be found in our [arxiv](-------) paper.

## Getting Started
**Checkpoints and Numbers:**

|                 | PS<br/>(ADE-150) | DE<br/>(NYU-V2) | OCR<br/>(6 datasets avg) | IC<br/>(VizWiz_Cap) | VQA<br/>(VizWiz_VQA) | #Params |
|-----------------|------------------|-----------------|--------------------------|---------------------|----------------------|---------|
| **Model**       | PQ               | RMSE            | Acc(%)                   | CIDEr               | Acc(%)               |         | 
| Unified-IO (S)  | -                | 0.649           | -                        | -                   | 42.4                 | 71M     | 
| Unified-IO (B)  | -                | 0.469           | -                        | -                   | 45.8                 | 241M    | 
| Unified-IO (L)  | -                | 0.402           | -                        | -                   | 47.7                 | 776M    | 
| X-Decoder (T)   | 41.6             | -               | -                        | -                   | -                    | 164M    | 
| GIT (T)         | -                | -               | -                        | 113.1               | 68.0                 | 0.7B    | 
| PaLI (T)        | -                |  -              | -                        | 117.2               | 67.5                 | 3.0B    | 
| [@Model](http:) | 38.5             | 0.425           | 80.1                     | 52.5                | 53.7                 | 62M     | 

**Installation, Dataset, Training and Evaluation Guide**
* [INSTALL.md](asset/INSTALL.md)
* [DATASET.md](asset/DATASET.md)
* [TRAIN.md](asset/TRAIN.md)
* [EVALUATION.md](asset/EVALUATION.md)

## Acknowledgement
* We build our work on top of [X-Decoder](https://github.com/microsoft/X-Decoder) and use their code. We appreciate the previous open-source repository [X-Decoder](https://github.com/microsoft/X-Decoder).

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{jiang2025atbench,
title={@BENCH: Benchmarking Vision-Language Models for Human-centered Assistive Technology},
author={Jiang, Xin and Zheng, Junwei and Liu, Ruiping and Li, Jiahang and Zhang, Jiaming and Matthiesen, Sven and Stiefelhagen, Rainer},
booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
year={2025}
}
```
