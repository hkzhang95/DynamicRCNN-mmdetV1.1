# Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training

## Introduction

```
@article{DynamicRCNN,
    author = {Hongkai Zhang and Hong Chang and Bingpeng Ma and Naiyan Wang and Xilin Chen},
    title = {Dynamic {R-CNN}: Towards High Quality Object Detection via Dynamic Training},
    journal = {arXiv preprint arXiv:2004.06002},
    year = {2020}
}
```

## Results and Models

| Model | Multi-scale training | AP (minival) | AP (test-dev) | Trained model |
|:---:|:---:|:---:|:---:|:---:|
| Dynamic_RCNN_r50_fpn_1x | No | 38.8 | - | - |
| Dynamic_RCNN_dcn_r101_fpn_3x | Yes | 46.8 | - | [Google Drive](https://drive.google.com/file/d/1AqJPCWb3qrNaeNRBnqh77vJspZuFTMkP/view?usp=sharing) |
| Dynamic_RCNN_dcn_r101_fpn_3x\* | Yes | - | 50.1 | [Google Drive](https://drive.google.com/file/d/1AqJPCWb3qrNaeNRBnqh77vJspZuFTMkP/view?usp=sharing) |

**Notes:**

1. `1x` and `3x` mean the model is trained for 90K and 270K iterations, respectively.
2. For `Multi-scale training`, the shorter side of images is randomly chosen from (400, 600, 800, 1000, 1200), and the longer side is 1400. To make a fair comparison with TridentNet, the training time is extended by `1.5x` under this setting (`2x` -> `3x`).
3. `dcn` denotes deformable convolutional networks v1 (no modulated version, applied to `c3-c5`).
4. `Dynamic_RCNN_dcn_r101_fpn_3x*` stands for `Dynamic R-CNN*` in Table 9 of the [paper](https://arxiv.org/abs/2004.06002). Compared to `Dynamic_RCNN_dcn_r101_fpn_3x`, it adds multi-scale testing (`shorter side`: [800, 1000, 1200, 1400, 1600], `longer side`: 2000) and Soft-NMS which can be easily modified in the same config file (with the commented lines).
5. This implementation is not compatible with the latest [mmdetection](https://github.com/open-mmlab/mmdetection) codebase. I will consider releasing a new version by making a pull request to the official codebase if I have time.
6. More trained models can be found at [this repo](https://github.com/hkzhang95/DynamicRCNN).
