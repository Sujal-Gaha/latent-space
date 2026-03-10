import torch
import torchvision

import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms


# Define the encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)

        return x_recon


# Example usage
input_dim = 784
latent_dim = 2
autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Load MNIST dataset
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(
    root="./600_outputs/from_blogs/codegenes",
    train=True,
    download=True,
    transform=transform,
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Training Loop

num_epochs = 500
for epoch in range(num_epochs):
    total_loss = 0
    for images, _ in trainloader:
        images = images.view(-1, 784)
        optimizer.zero_grad()
        reconstructed = autoencoder(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(trainloader):.4f}")

latent_representations = []
labels = []

with torch.no_grad():
    for data in trainloader:
        images, batch_labels = data
        images = images.view(-1, 784)
        z = autoencoder.encoder(images)
        latent_representations.extend(z.numpy())
        labels.extend(batch_labels.numpy())

latent_representations = np.array(latent_representations)
labels = np.array(labels)


# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(
    latent_representations[:, 0], latent_representations[:, 1], c=labels, cmap="viridis"
)
plt.colorbar()
plt.title("Latent Space Visualization of MNIST")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()
