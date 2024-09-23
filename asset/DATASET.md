# Preparing Dataset

ATModel can perform 5 tasks simultaneously, namely Panoptic Segmentation, Captioning, VQA, Depth Estimation and OCR.
Below are the details of the datasets for different tasks.

```sh
Panoptic Segmentation -> ADE20K
Captioning -> VizWiz_Cap
VQA -> VizWiz_VQA
Depth Estimation -> NYU_v2
OCR -> MJSynth(MJ), SynthText (ST), ICDAR_2013(IC13), ICDAR_2015 (IC15), IIIT5K-Words (IIIT5K), Street View Text (SVT), Street ViewText-Perspective (SVTP), CUTE80(CUTE)
```


## Panoptic Segmentation 
Expected dataset structure for [ADE20K](http://sceneparsing.csail.mit.edu/)
```
.atmodel_data/
└── seg_datatsets/ADEChallengeData2016/
    └── images/
        └── training/
           ├── ADE_train_00000001.jpg
           ├── ...
        └── validation/
           ├── ADE_val_00000001.jpg 
           ├── ...
    └── ade20k_panoptic_train/
        ├── ADE_train_00000001.png 
        ├── ...
    └── ade20k_panoptic_val/
        ├── ADE_val_00000001.png 
        ├── ...
    ├── ade20k_panoptic_train.json
    ├── ade20k_panoptic_val.json
```

## Captioning
Expected dataset structure for [VizWiz_Cap](https://vizwiz.org/tasks-and-datasets/image-captioning/)
```
.atmodel_data/
└── captioning_datasets/vizwiz/
    └── train/
        ├── VizWiz_train_00000000.jpg
        ├── ...
    └── val/
        ├── VizWiz_val_00000000.jpg
        ├── ...
    └── test/
        ├── VizWiz_test_00000000.jpg
        ├── ...
    └── annotations/
       ├── train.json
       ├── val.json
       ├── test.json
```

## VQA
Expected dataset structure for [VizWiz_VQA](https://vizwiz.org/tasks-and-datasets/vqa/)
```
.atmodel_data/
└── vqa_datasets/vizwiz/
    └── train/
        ├── VizWiz_train_00000000.jpg
        ├── ...
    └── val/
        ├── VizWiz_val_00000000.jpg
        ├── ...
    └── test/
        ├── VizWiz_test_00000000.jpg
        ├── ...
    └── annotations/
       ├── train.json
       ├── val.json
       ├── test.json
```

## Depth Estimation
Please follow the [BTS](https://github.com/cleinc/bts/tree/master/pytorch) to download the NYU_v2 dataset and prepare the dataset as below.
```
.atmodel_data/
└── captioning_datasets/vizwiz/
    └── images/
        ├── 0.jpg
        ├── ...
    └── raw_depths/
        ├── 0.png
        ├── ...
    ├── train.txt
    ├── val.txt
```

## OCR
Please follow the [ABINet](https://github.com/FangShancheng/ABINet) to download the OCR datasets and prepare the datasets as below.
```
.atmodel_data/
└── ocr_datasets
    └── training/
        └── MJ/
            └── MJ_train/
            └── MJ_valid/
            └── MJ_test/
        └── ST/
    └── evaluation/
        └── IIIT5k_3000/
        └── SVT/
        └── SVTP/
        └── IC13_857/
        └── IC15_1811/
        └── CUTE80
```



