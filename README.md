# ScribbleLight: Single Image Indoor Relighting with Scribbles (CVPR 2025)

  <p align="center">
    <a href="https://chedgekorea.github.io"><strong>Jun Myeong Choi</strong></a>
    ·    
    <strong>Annie Wang</strong>
    ·
    <a href="https://www.cs.wm.edu/~ppeers/"><strong>Pieter Peers</strong></a>
    ·
    <a href="https://anandbhattad.github.io"><strong>Anand Bhattad</strong></a>
    ·
    <a href="https://www.cs.unc.edu/~ronisen/"><strong>Roni Sengupta</strong></a>
  </p>   
  <p align="center">
    <a href="https://chedgekorea.github.io/ScribbleLight/"><strong>Project Page</strong></a>
    |    
    <a href="https://arxiv.org/abs/2411.17696"><strong>Paper</strong></a>

  </p> 

## :book: Abstract

Image-based relighting of indoor rooms creates an immersive virtual understanding of the space, which is useful for interior design, virtual staging, and real estate. Relighting indoor rooms from a single image is especially challenging due to complex illumination interactions between multiple lights and cluttered objects featuring a large variety in geometrical and material complexity. Recently, generative models have been successfully applied to image-based relighting conditioned on a target image or a latent code, albeit without detailed local lighting control. In this paper, we introduce ScribbleLight, a generative model that supports local fine-grained control of lighting effects through scribbles that describe changes in lighting. Our key technical novelty is an Albedo-conditioned Stable Image Diffusion model that preserves the intrinsic color and texture of the original image after relighting and an encoder-decoder-based ControlNet architecture that enables geometry-preserving lighting effects with normal map and scribble annotations. We demonstrate ScribbleLight's ability to create different lighting effects (e.g., turning lights on/off, adding highlights, cast shadows, or indirect lighting from unseen lights) from sparse scribble annotations.

---

## :wrench: Setup

please install PyTorch using the following command:

```
conda create -n scriblit python=3.9
conda activate scriblit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

The exact versioning may vary depending on your computing environment and what GPUs you have access to. Note this <a href="https://towardsdatascience.com/managing-multiple-cuda-versions-on-a-single-machine-a-comprehensive-guide-97db1b22acdc/">good article</a> for maintaining multiple system-level versions of CUDA.

Download <a href="https://github.com/huggingface/diffusers">diffusers</a> using the following code.

```
conda install -c conda-forge diffusers
```

---

## dataset

The data is organized inside the dataset folder as follows:
Using the target image as input: normal is obtained using <a href="https://github.com/baegwangbin/DSINE">DSINE</a>, shading and albedo are obtained using <a href="https://github.com/compphoto/Intrinsic">IID</a>, and the prompt is generated using <a href="https://github.com/salesforce/LAVIS/tree/main/projects/blip2">BLIP-2</a>. The image paths and prompts should be saved in the dataset/data/prompt.json file.
(Note that during training, the shading map is used, but during inference, the user scribble is used instead of the shading map.)
```
$ROOT/dataset
└── data
   └── normal
   └── shading
   └── albedo
   └── target
   └── prompt
   └── prompt.json 
```
---

## :computer: Training

STEP1: Train Stable Diffusion

    ```
    export MODEL_DIR="stabilityai/stable-diffusion-2-1"
    export OUTPUT_DIR=scribblelight_stable_diffusion

    accelerate launch train_stable_diffusion.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --validation_image "./imnormal_1.png" "./imnormal_1.png" \
    --validation_prompt "a porch with a bed chairs and a table" "a porch with a bed chairs and a table" \
    --dataset=data \
    ```
    
STEP2: Train ControlNet

    ```
    export MODEL_DIR="stabilityai/stable-diffusion-2-1"
    export OUTPUT_DIR=scribblelight_controlnet
    
    accelerate launch train_controlnet.py \
     --pretrained_model_name_or_path=$MODEL_DIR \
     --output_dir=$OUTPUT_DIR \
     --pretrain_unet_path=scribblelight_stable_diffusion/checkpoint-41000 \
     --validation_image "./imnormal_1.png" "./imnormal_1.png" \
     --validation_prompt "a porch with a bed chairs and a table" "a porch with a bed chairs and a table" \
     --dataset=data
    ```

## :computer: Evaluation

STEP1: Please download the pretrained weights.
STEP2: Generate the predictions.
    ```
    CUDA_VISIBLE_DEVICES=0 python eval.py --test_data_root datasets/THuman/val --regressor_path weights/model_gaussian.pth --inpaintor_path weights/model_inpaint.pth
    ```
   The results will be saved at `$ROOT/inference/{$data}`.

## :scroll: Citation
If you find this code useful for your research, please cite it using the following BibTeX entry.

```
@misc{choi2024scribblelightsingleimageindoor,
		title={ScribbleLight: Single Image Indoor Relighting with Scribbles}, 
		author={Jun Myeong Choi and Annie Wang and Pieter Peers and Anand Bhattad and Roni Sengupta},
		year={2024},
		eprint={2411.17696},
		archivePrefix={arXiv},
		primaryClass={cs.CV},
		url={https://arxiv.org/abs/2411.17696}, 
  }
```
