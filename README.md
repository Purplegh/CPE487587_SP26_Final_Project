# CPE487587_SP26_Final_Project
# Malaria Cell Image Compression and Super-Resolution

## Dataset

The dataset used for this project is the **Cell Images for Detecting Malaria** dataset. It consists of microscopic blood smear images of red blood cells, each labeled as either Parasitized (infected with malaria) or Uninfected (healthy). The images were captured using Giemsa-stained thin blood smear slides under a microscope.

- **Total Samples:** 27,558 cell images
- **Parasitized:** 13,779 images
- **Uninfected:** 13,779 images

## Features

Since this is an image dataset, each sample is a PNG microscopy image with the following properties:

- **Raw Pixel Values:** Each image is an RGB array of shape (H x W x 3) with pixel values in range [0, 255]
- **Cell Morphology:** Infected cells show dark purple ring-form trophozoites visible under Giemsa staining
- **Color Distribution:** Parasitized cells have a different color histogram due to the parasite occupying space inside the cell
- **Texture:** The granular surface texture differs between infected and healthy cells
- **Cell Shape:** Parasitized cells may show irregular boundaries compared to smooth healthy RBCs
- **Image Resolution:** Variable, approximately 100x100 to 385x395 pixels
- **Format:** PNG, RGB (3 channels)

## Labels

| Label | Description |
|---|---|
| `Parasitized` | Red blood cell infected with *Plasmodium* malaria parasite |
| `Uninfected` | Healthy red blood cell with no infection |

## Problem Statement

In low-resource clinical settings, high-resolution microscopy images are difficult to store and transmit due to limited bandwidth and storage infrastructure. This project addresses the problem of **medical image compression and high-quality reconstruction** using two stages:

1. **Dimensionality Reduction via Convolutional Autoencoder:** Compresses the original cell images into a compact latent representation, reducing data size while retaining diagnostically relevant features.
2. **Super-Resolution via Denoising GAN:** The decoded output from the autoencoder is passed through a GAN to recover fine-grained visual details, sharpen boundaries, and restore textures lost during compression.

## How to Obtain the Dataset

**Via TensorFlow Datasets:**
```python
pip install tensorflow-datasets
python -c "import tensorflow_datasets as tfds; tfds.load('malaria', download=True, data_dir='./data')"
```

**Kaggle (alternative):** https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

**Dataset Size:** ~337 MB — suitable for download directly to the Lovelace account.

## Original Source

Originally curated and published by the **U.S. National Institutes of Health (NIH), National Library of Medicine (NLM)**.

> NIH Official Page: https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html

## Setup
 
Clone the repository and navigate to the project directory:
 
```bash
git clone https://github.com/Purplegh/CPE487587_SP26_Final_Project
cd CPE487587_SP26_Final_Project
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
```
 
Install dependencies:
 
```bash
uv pip install torch torchvision scikit-image matplotlib numpy pillow
```


 ## Data Directory Structure

Before running the pipeline, ensure your dataset is organized as follows inside the `data/` folder at the root of the project directory:
```
data/
├── train/        # 22,050 training images (.png)
├── val/          # 2,754 validation images (.png)
└── test/         # 2,754 test images (.png)
```
---
 
## Running the Pipeline
 
The pipeline consists of three scripts: `autoencoder.py`, `gan.py`, and `evaluate.py`. Each script accepts a run number (1, 2, or 3) corresponding to different random seeds for reproducibility. All three runs must be completed before generating the final summary.
 
### Step 1 — Train the Autoencoder
 
```bash
nohup python autoencoder.py 1 > logs/run1_ae.out 2>&1 &
nohup python autoencoder.py 2 > logs/run2_ae.out 2>&1 &
nohup python autoencoder.py 3 > logs/run3_ae.out 2>&1 &
```
 
This trains the convolutional autoencoder for each run and saves:
- `weights/run{N}/autoencoder.pt` — trained autoencoder weights
- `weights/run{N}/encoder.onnx` — ONNX encoder 
- `weights/run{N}/decoder.onnx` — ONNX decoder 
- `results/run{N}/autoencoder_results.png` — original vs reconstructed figure
- `logs/run{N}/autoencoder.log` — training loss log
### Step 2 — Train the GAN
 
> **Note:** Run this only after the autoencoder for the corresponding run has finished training.
 
```bash
nohup python gan.py 1 > logs/run1_gan.out 2>&1 &
nohup python gan.py 2 > logs/run2_gan.out 2>&1 &
nohup python gan.py 3 > logs/run3_gan.out 2>&1 &
```
 
This loads the frozen autoencoder and trains the GAN generator for each run and saves:
- `weights/run{N}/generator.pt` — trained GAN generator weights
- `weights/run{N}/discriminator.pt` — trained GAN discriminator weights
- `results/run{N}/gan_results.png` — original vs AE vs GAN figure
- `logs/run{N}/gan.log` — training loss log
### Step 3 — Evaluate
 
> **Note:** Run this only after both the autoencoder and GAN for the corresponding run have finished training.
 
```bash
nohup python evaluate.py 1 > logs/run1_eval.out 2>&1 &
nohup python evaluate.py 2 > logs/run2_eval.out 2>&1 &
nohup python evaluate.py 3 > logs/run3_eval.out 2>&1 &
```
 
This runs inference on the full test set and saves:
- `results/run{N}/metrics.txt` — per-run PSNR, SSIM, VoL, TEN metrics
- `results/run{N}/inference_results.png` — final inference figure
### Step 4 — Generate Final Summary
 
> **Note:** Run this only after all 3 runs of evaluate.py have completed.
 
```bash
python evaluate.py all
```
 
This reads `metrics.txt` from all three runs and prints the mean ± standard deviation across runs for all four metrics. The summary is saved to:
- `results/summary.txt`
---