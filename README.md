# 3D Face Reconstruction with StyleGAN3-based Texture Synthesis from Multi-View Images

---
> 3D model generation from a single image is a challenging task due to the lack of
texture information and limited training data. This model proposes a novel approach
for texture estimation from a single image using a generative adversarial network
(StyleGAN3) and 3D Dense Face Alignment (3DDFA). 
The method begins by generating multi-view faces using the latent space of StyleGAN3 using Restyle encoder. 
Then 3DDFA generate a high-resolution texture map and map it to 3D model that is consistent with the estimated
face shape.

[![GitHub license](https://img.shields.io/github/license/rohit7044/3DGANTex)](https://github.com/rohit7044/3DGANTex/blob/main/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/rohit7044/3DGANTex/graphs/commit-activity)

[![Stargazers repo roster for @rohit7044/3DGANTex](https://reporoster.com/stars/rohit7044/3DGANTex)](https://github.com/rohit7044/3DGANTex/stargazers)

[![Forkers repo roster for @rohit7044/3DGANTex](https://reporoster.com/forks/rohit7044/3DGANTex)](https://github.com/rohit7044/3DGANTex/network/members)
## White paper coming soon!
<!-- ![workflow](https://lucid.app/publicSegments/view/8ff49995-cb25-47ae-9dca-5aa88caee4a9/image.jpeg) -->

## Requirements
1. Ubuntu 22.04
2. Python 3.8
3. PyTorch(2.0 works great)
4. OpenCV
5. Dlib
6. Cython
7. Cmake

## StyleGAN3 Encoder
Download the pretrained encoder from the following links and keep it on `pretrained_model` folder

| Encoder                     | Description                                                               |
|-----------------------------|---------------------------------------------------------------------------|
| [ReStyle-pSp Human Faces](https://drive.google.com/file/d/12WZi2a9ORVg-j6d9x4eF-CKpLaURC2W-/view?usp=sharing) | ReStyle-pSp trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset over the StyleGAN3 generator. |
| [ReStyle-e4e Human Faces](https://drive.google.com/file/d/1z_cB187QOc6aqVBdLvYvBjoc93-_EuRm/view) | ReStyle-e4e trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset over the StyleGAN3 generator. |

## Usage
1. Clone the Repo:

~~~ 
git clone https://github.com/rohit7044/3DGANTex 
~~~
2. Download both the pretrained models mentioned above
3. Build the cython version of NMS, Sim3DR, and the faster mesh render on the main directory 
~~~
sh ./TDDFA_build.sh
~~~
4. Open `3D-GANTex.py` and make the changes mentioned on the code inside
5. Finally after making the changes, upon running the code you will get multi_view,texture map and 3d model and it will be saved in `output_data`

## Special Thanks
* [Third Time's the Charm? Image and Video Editing with StyleGAN3](https://github.com/yuval-alaluf/stylegan3-editing)

* [InterFaceGAN - Interpreting the Latent Space of GANs for Semantic Face Editing](https://github.com/genforce/interfacegan)

* [3DDFA_V2 Towards Fast, Accurate and Stable 3D Dense Face Alignment](https://github.com/cleardusk/3DDFA_V2)

## Important Note
1. The 3D face model has uv texture embedded but it only shows the texture on Meshlab and Open3D
2. Weak results on images with glasses.
3. Better to take portrait image that has only the face like the example mentioned in `input_data` directory.
