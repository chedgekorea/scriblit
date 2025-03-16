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

The data is organized inside the dataset folder as follows. Using the target image as input: normal is obtained using <a href="https://github.com/baegwangbin/DSINE">DSINE</a>, shading and albedo are obtained using <a href="https://github.com/compphoto/Intrinsic">IID</a>, and the prompt is generated using <a href="https://github.com/salesforce/LAVIS/tree/main/projects/blip2">BLIP-2</a>. The image paths and prompts should be saved in the dataset/data/prompt.json file.
(Note that during training, the shading map is used, but during inference, the user scribble in the shading map folder should be used instead.)
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

## Training
    ```
    CUDA_VISIBLE_DEVICES=0 python train_nightly_ver.py
    ```

## Evaluation

We provide detailed information about the evaluation protocol in [PROTOCOL.md](PROTOCOL.md).
To make the comparison with our Generalizable Human Gaussians easier, we provide the evaluation results in this [link](https://1drv.ms/u/s!Aq9xVNM_DjPG5RDhuwsv4XaaP62v?e=acMiPv).

1. Please download the pretrained weights following the instructions in [INSTALL.md](INSTALL.md).
2. Generate the predictions.
    ```
    CUDA_VISIBLE_DEVICES=0 python eval.py --test_data_root datasets/THuman/val --regressor_path weights/model_gaussian.pth --inpaintor_path weights/model_inpaint.pth
    ```
   The results will be saved at `$ROOT/outputs/eval/{$exp_name}`.
3. Compute the metrics.
    ```
    python metrics/compute_metrics.py
    ``` 

## Citation

If you find this code useful for your research, please cite it using the following BibTeX entry.

```
@article{kwon2024ghg,
  title={Generalizable Human Gaussians for Sparse View Synthesis},
  author={Youngjoong Kwon, Baole Fang, Yixing Lu, Haoye Dong, Cheng Zhang, Francisco Vicente Carrasco, Albert Mosella-Montoro, Jianjin Xu, Shingo Takagi, Daeil Kim, Aayush Prakash, Fernando De la Torre},
  journal={European Conference on Computer Vision},
  year={2024}
}
```
