
# Synthetic Sniffer

## Overview
Synthetic Sniffer is a project aimed at detecting synthetic faces generated by various GAN models using a hierarchical disentanglement ensemble (HDE). The HDE model integrates multiple pathways to analyze different aspects of the input images and combines their outputs to classify real vs. fake images.

## Project Structure
- `FYP/HDE_FYP.ipynb`: Notebook for training the HDE model using StyleGAN & Flickr datasets.
- `FYP/HDE_FYP_TEST_PROGAN.ipynb`: Notebook for testing the HDE model using the PRO-GAN dataset.

## Datasets
The project utilizes datasets from Kaggle:
1. **Training and Validation**: StyleGAN & Flickr datasets
2. **Testing**: PRO-GAN dataset

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sevinpr/Synthetic-Sniffer.git
   cd Synthetic-Sniffer
 

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have Kaggle API credentials set up to download the datasets.

## Usage
### Training the Model
1. Open the `FYP/HDE_FYP.ipynb` notebook.
2. Run the cells to download the dataset, preprocess the data, and train the HDE model.

### Testing the Model
1. Open the `FYP/HDE_FYP_TEST_PROGAN.ipynb` notebook.
2. Run the cells to download the test dataset, preprocess the data, and evaluate the HDE model.

## Model Architecture
### Hierarchical Disentanglement Ensemble (HDE)
- **First-Level Disentanglement**: Implement a DR-GAN (Disentangled Representation learning-Generative Adversarial Network) framework to learn identity representations that are explicitly disentangled from pose and other variations. This step provides a foundation by separating identity-related features from content-related features.
- **Second-Level Analysis**: Incorporate ForensicsForest Family's multi-scale hierarchical cascade forest approach to analyze the disentangled representations rather than raw images. The analysis is carried out through semantic, frequency, and biological feature extraction pathways.
- **Contrastive Refinement Layer**: Add a contrastive disentanglement module that uses negative-free contrastive learning to further separate authentic identity factors from GAN artifacts. This enhances the detection of subtle inconsistencies.
- **Adversarial Equilibrium Component**: Integrate a NashAE-inspired component to promote a sparse and disentangled latent space, mitigating evasion attempts by ensuring that the latent factors remain robust and distinct.
- **Decision Fusion Mechanism**: Instead of simple voting or averaging, implement a dynamic weighting scheme based on the Total AUROC Difference (TAD) metric. This mechanism dynamically fuses the outputs from all components, improving the overall reliability of the detection process.

### Training and Evaluation
The training process involves:
- Classification loss from both DR-GAN and fusion outputs.
- Reconstruction loss from DR-GAN and NashAE modules.
- Contrastive loss from the contrastive module.
- Adversarial loss from the NashAE discriminator.

Evaluation metrics include accuracy, precision, recall, F1 score, and AUC.

## Results
The model is trained for 10 epochs, showing improvements in validation accuracy and other metrics. Here are the results from the training:
- **Final Train Loss**: 1.0043
- **Final Validation Metrics**: 
  - Loss: 0.2474
  - Accuracy: 0.8965
  - Precision: 0.8989
  - Recall: 0.8965
  - F1 Score: 0.8963
  - AUC: 0.8965

## Contributing
Feel free to open issues or submit pull requests if you have suggestions or improvements for the project.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Kaggle for providing the datasets.
- Authors and maintainers of the pre-trained models used in this project.
```
