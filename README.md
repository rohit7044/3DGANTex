# 3D Face Reconstruction with StyleGAN3-based Texture Synthesis from Multi-View Images

---

[![GitHub license](https://img.shields.io/github/license/rohit7044/3DGANTex)](https://github.com/rohit7044/3DGANTex/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/rohit7044/3DGANTex)](https://github.com/rohit7044/3DGANTex/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/rohit7044/3DGANTex)](https://github.com/rohit7044/3DGANTex/issues)
[![GitHub forks](https://img.shields.io/github/forks/rohit7044/3DGANTex)](https://github.com/rohit7044/3DGANTex/network)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/rohit7044/3DGANTex/graphs/commit-activity)
## White paper soming soon!!!

![workflow](https://lucid.app/publicSegments/view/8ff49995-cb25-47ae-9dca-5aa88caee4a9/image.jpeg)

## Requirements
1. Python 3.8
2. PyTorch(2.0 works great)
3. OpenCV
4. Dlib

## StyleGAN3 Encoder
Download the pretrained encoder from the following links and keep it on ``pret``

| Encoder                     | Description                                                               |
|-----------------------------|---------------------------------------------------------------------------|
| [ReStyle-pSp Human Faces](https://drive.google.com/file/d/12WZi2a9ORVg-j6d9x4eF-CKpLaURC2W-/view?usp=sharing) | ReStyle-pSp trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset over the StyleGAN3 generator. |
| [ReStyle-e4e Human Faces](https://drive.google.com/file/d/1z_cB187QOc6aqVBdLvYvBjoc93-_EuRm/view) | ReStyle-e4e trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset over the StyleGAN3 generator. |

