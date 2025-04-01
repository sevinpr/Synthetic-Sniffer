#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAN-Generated Image Detection using Disentangled Features

This script implements a PyTorch model to detect if an image is GAN-generated or not,
using disentangled features extracted from the images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE


class DisentangledFeatureExtractor(nn.Module):
    """
    Feature extractor that disentangles content and style features from images.
    Uses a pre-trained ResNet50 as the backbone.
    """
    def __init__(self, pretrained=True, num_features=256):
        super(DisentangledFeatureExtractor, self).__init__()
        # Use a pre-trained model as the backbone
        self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Remove the classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add disentangled feature extraction layers
        self.content_encoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_features//2)
        )
        
        self.style_encoder = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_features//2)
        )
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Disentangle into content and style features
        content_features = self.content_encoder(features)
        style_features = self.style_encoder(features)
        
        # Concatenate disentangled features
        disentangled_features = torch.cat([content_features, style_features], dim=1)
        
        return disentangled_features


class GANDetector(nn.Module):
    """
    Classifier that predicts if an image is GAN-generated based on disentangled features.
    """
    def __init__(self, feature_dim=256):
        super(GANDetector, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        return self.classifier(features)


class GANDataset(Dataset):
    """
    Custom dataset for loading real and GAN-generated images.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def prepare_data(real_dir, gan_dir, batch_size=32):
    """
    Prepare the datasets and dataloaders.
    
    Args:
        real_dir: Directory containing real images
        gan_dir: Directory containing GAN-generated images
        batch_size: Batch size for the dataloaders
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get image paths
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    gan_images = [os.path.join(gan_dir, f) for f in os.listdir(gan_dir) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    all_images = real_images + gan_images
    labels = [0] * len(real_images) + [1] * len(gan_images)  # 0 for real, 1 for GAN
    
    # Split into train, validation, and test sets
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        all_images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        train_imgs, train_labels, test_size=0.25, random_state=42, stratify=train_labels
    )
    
    # Create datasets
    train_dataset = GANDataset(train_imgs, train_labels, transform)
    val_dataset = GANDataset(val_imgs, val_labels, transform)
    test_dataset = GANDataset(test_imgs, test_labels, transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def train_epoch(feature_extractor, classifier, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    feature_extractor.train()
    classifier.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        # Extract disentangled features
        features = feature_extractor(images)
        
        # Predict using classifier
        outputs = classifier(features)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(feature_extractor, classifier, dataloader, criterion, device):
    """Validate the model"""
    feature_extractor.eval()
    classifier.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            features = feature_extractor(images)
            outputs = classifier(features)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc, all_preds, all_labels


def plot_training_history(history):
    """Plot training and validation loss and accuracy"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def visualize_features(feature_extractor, dataloader, device):
    """Visualize disentangled features using t-SNE"""
    feature_extractor.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            features = feature_extractor(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    # Use t-SNE for dimensionality reduction
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[all_labels==0, 0], features_2d[all_labels==0, 1], 
                c='blue', label='Real', alpha=0.5)
    plt.scatter(features_2d[all_labels==1, 0], features_2d[all_labels==1, 1], 
                c='red', label='GAN', alpha=0.5)
    plt.legend()
    plt.title('t-SNE Visualization of Disentangled Features')
    plt.savefig('feature_visualization.png')
    plt.show()
    
    return all_features, all_labels


def predict_image(image_path, feature_extractor, classifier, device):
    """Predict if a single image is GAN-generated or real"""
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract features and make prediction
    feature_extractor.eval()
    classifier.eval()
    
    with torch.no_grad():
        features = feature_extractor(image_tensor)
        output = classifier(features)
        
        prob = output.item()
        prediction = "GAN-generated" if prob > 0.5 else "Real"
        confidence = prob if prob > 0.5 else 1 - prob
    
    # Display the image and prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(image))
    plt.title(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
    plt.axis('off')
    plt.show()
    
    return prediction, confidence


def train_model(real_dir, gan_dir, num_epochs=20, batch_size=32, learning_rate=0.0001):
    """Train and evaluate the GAN detector model"""
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(real_dir, gan_dir, batch_size)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    feature_extractor = DisentangledFeatureExtractor().to(device)
    classifier = GANDetector().to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    
    # Combine parameters from both models for optimization
    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + list(classifier.parameters()), 
        lr=learning_rate, 
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            feature_extractor, classifier, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            feature_extractor, classifier, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch
            }, 'best_gan_detector_model.pth')
            print("Saved best model checkpoint")
    
    # Test best model
    print("\nLoading best model for testing...")
    checkpoint = torch.load('best_gan_detector_model.pth')
    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels = validate(
        feature_extractor, classifier, test_loader, criterion, device
    )
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        np.array(test_labels).flatten(), 
        np.array(test_preds).flatten(),
        target_names=['Real', 'GAN']
    ))
    
    # Visualize confusion matrix
    cm = confusion_matrix(np.array(test_labels).flatten(), np.array(test_preds).flatten())
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Real', 'GAN'])
    plt.yticks(tick_marks, ['Real', 'GAN'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Plot training history
    plot_training_history(history)
    
    # Visualize features
    print("\nVisualizing disentangled features...")
    visualize_features(feature_extractor, test_loader, device)
    
    return feature_extractor, classifier, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GAN-Generated Image Detection using Disentangled Features")
    parser.add_argument('--real_dir', required=True, help='Directory containing real images')
    parser.add_argument('--gan_dir', required=True, help='Directory containing GAN-generated images')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--predict', help='Path to an image for prediction (optional)')
    
    args = parser.parse_args()
    
    if args.predict:
        if not os.path.exists('best_gan_detector_model.pth'):
            print("Error: Model checkpoint not found. Train the model first.")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            feature_extractor = DisentangledFeatureExtractor().to(device)
            classifier = GANDetector().to(device)
            
            checkpoint = torch.load('best_gan_detector_model.pth')
            feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            classifier.load_state_dict(checkpoint['classifier'])
            
            prediction, confidence = predict_image(args.predict, feature_extractor, classifier, device)
            print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
    else:
        # Train the model
        feature_extractor, classifier, history = train_model(
            args.real_dir, args.gan_dir, 
            num_epochs=args.epochs, 
            batch_size=args.batch_size,
            learning_rate=args.lr
        ) 