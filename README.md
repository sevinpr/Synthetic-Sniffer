# Hybrid VAE with Disentangled Features for GAN Image Detection

This repository contains an end-to-end implementation of a hybrid variational autoencoder (VAE) that integrates a classification head and leverages triplet loss to enforce latent space separability. By learning disentangled features, the model enhances its ability to distinguish between real images and GAN-generated images. The code is optimized for execution in Google Colab, including mounting Google Drive for data access and leveraging GPU acceleration when available.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [Setup and Usage](#setup-and-usage)
  - [1. Mounting Google Drive & Data Preparation](#1-mounting-google-drive--data-preparation)
  - [2. Data Loading & Preprocessing](#2-data-loading--preprocessing)
  - [3. Model Architecture](#3-model-architecture)
  - [4. Loss Functions](#4-loss-functions)
  - [5. Training the Model](#5-training-the-model)
  - [6. Evaluation & Visualization](#6-evaluation--visualization)
  - [7. Predicting on New Images](#7-predicting-on-new-images)
- [Future Directions](#future-directions)
- [Conclusion](#conclusion)

---

## Overview

The project implements a hybrid VAE model that simultaneously performs:
- **Image Reconstruction:** Learning a compressed representation of input images.
- **Classification:** Differentiating between real images and GAN-generated images using a dedicated classification head.
- **Disentangled Feature Learning:** Leveraging disentangled latent representations to better separate the underlying factors of variation, which enhances the detection of GAN-generated images.
- **Latent Space Optimization:** Enforcing separability via a triplet loss function to make the latent space more discriminative.

This approach not only facilitates the reconstruction and generation of images but also significantly improves classification performance by extracting and utilizing disentangled features.

---

## Features

- **Hybrid VAE Model:** Combines a conventional VAE with a classifier to enable joint learning of image reconstruction and GAN image detection.
- **Disentangled Feature Extraction:** Learns separate, meaningful latent factors that improve the distinction between real and GAN-generated images.
- **Triplet Loss Integration:** Enhances latent feature separability by penalizing insufficient separation between classes.
- **t‑SNE Visualization:** Provides intuitive 2D visualizations of the disentangled latent space, allowing assessment of class clustering.
- **Google Colab Compatibility:** Designed to run seamlessly in a Colab environment with support for GPU acceleration and Google Drive integration.

---

## Requirements

- **Python 3.x**
- **PyTorch**
- **Torchvision**
- **scikit-learn**
- **matplotlib**
- **Pillow (PIL)**
- **Google Colab Environment** (recommended for GPU support and easy file management)

Install the required packages using:
```bash
pip install torch torchvision scikit-learn matplotlib pillow
