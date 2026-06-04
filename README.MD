# Counterfactual Chest X-Ray Generation via Disentangled Anatomy and Diffusion Transformers (SiT)

[License: MIT](https://opensource.org/licenses/MIT) | [Python 3.10+](https://www.python.org/downloads/) | [PyTorch](https://pytorch.org/)

This repository provides the official PyTorch implementation for generating counterfactual disease trajectories (progression and regression) in Chest X-rays. By explicitly disentangling stable anatomical structures from transient pathological features, this framework allows for the continuous manipulation of disease severity (e.g., Cardiomegaly, Pneumonia) while strictly preserving patient identity.

## Overview

Generative modeling in medical imaging often struggles with identity preservation during image translation. This repository solves this by pairing a domain-specific Anatomy Encoder with a Scalable Interpolation Transformer (SiT). 

Our pipeline enables:
* Continuous Severity Dials: Interpolate disease severity smoothly from 0.0 (Healthy) to 1.0 (Severe Pathology).
* Counterfactual Regression/Progression: Simulate how a specific patient's X-ray would look if their condition worsened or improved.
* Identity Preservation: Zero-shot retention of unique patient rib cages, lung shapes, and clavicles using orthogonal spatial latents.

## Methodology

1. Disentangled Anatomy Encoder:
We utilize a custom segmentation head (supervised via TorchXRayVision pseudo-masks) and an adversarial pathology probe to force the encoder to learn only structural anatomy. The resulting latent vector is orthogonal to disease presence.

2. SiT Diffusion Wrapper:
A Diffusion Transformer (SiT-XL/2) is trained to denoise X-ray latents, conditioned jointly on the disentangled anatomy tokens and a continuous scalar condition representing disease severity.

## Installation

```bash
git clone [https://github.com/yourusername/counterfactual-cxr-sit.git](https://github.com/yourusername/counterfactual-cxr-sit.git)
cd counterfactual-cxr-sit

# Create and activate environment
conda create -n counterfactual python=3.9
conda activate counterfactual

# Install dependencies
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt
```

## Data Preparation

The model is trained on frontal (PA / AP) views from the MIMIC-CXR dataset. Create a manifest CSV with the following required columns:
`dicom_id`, `condition`, `our_split` (train/val), and `filepath` (absolute path to the image).

## Training Pipeline

### 1. Train the Anatomy Encoder
The encoder learns to isolate patient anatomy (Background, Lungs, Heart, Central Silhouette) from pathology.
```bash
python train_encoder.py --manifest path/to/manifest.csv --epochs 50 --batch_size 16
```
*Outputs: `checkpoints/best_anatomy_encoder_10k.pt`*

### 2. Train the Diffusion Model (SiT)
Trains the SiT model conditioned on the locked spatial tokens from the Anatomy Encoder.
```bash
python train_diffusion.py --manifest path/to/manifest.csv --anatomy_ckpt checkpoints/best_anatomy_encoder_10k.pt
```
*Outputs: `checkpoints/best_checkpoint.pt`*

## Inference and GUI

We provide a local GUI (Tkinter) to visualize disease timelines interactively.

```bash
python app.py
```
Features in the GUI:
* Upload raw patient X-rays.
* Toggle between Progression (Healthy -> Sick) and Regression (Sick -> Healthy).
* Adjust Edit Strength and Classifier-Free Guidance (CFG) scales.
* Export high-resolution severity grids.

## Citation

If you use this code or model in your research, please cite our work:

```bibtex
@article{yourName2024counterfactual,
  title={Counterfactual Chest X-Ray Generation via Disentangled Anatomy and Diffusion Transformers},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Acknowledgments
* Segmentations derived using TorchXRayVision.
* Core diffusion architectures adapted from SiT and Stable Diffusion.
