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