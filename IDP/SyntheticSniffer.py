import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Model Definition
# ---------------------------
LATENT_DIM = 24

class HybridVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=2):
        super(HybridVAE, self).__init__()
        # Encoder: 3 x 128 x 128 -> feature map
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU()
        )
        # Fully connected layers for latent space
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        # Decoder: map latent vector back to feature maps and then upsample
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 64x64 -> 128x128
            nn.Tanh()  # outputs in [-1, 1]
        )
        # Classifier head based on the latent mean
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten the feature maps
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 8, 8)  # Reshape into feature maps
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        logits = self.classifier(mu)
        return x_recon, mu, logvar, logits

# ---------------------------
# Model Loading using st.cache_resource
# ---------------------------
@st.cache_resource
def load_model(model_path="hybrid_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridVAE(latent_dim=LATENT_DIM, num_classes=2)
    # Ensure the model file "hybrid_model.pth" is in the same directory or adjust the path accordingly.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ---------------------------
# Image Preprocessing
# ---------------------------
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Sythetic Sniffer")
st.caption("Detecting GAN Generated images using Disentangled Features")
st.write("Upload an image, and the model will predict whether it is fake or real.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # add batch dimension
    
    if st.button("Predict"):
        with torch.no_grad():
            _, _, _, logits = model(image_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            # Mapping: 0 = Fake, 1 = Real
            label = "Real" if pred_idx == 1 else "Fake"
            st.success(f"Prediction: {label}")
