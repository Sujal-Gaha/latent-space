## What This Code Does (Big Picture)

It builds a neural network that learns to **compress** handwritten digit images from 784 numbers down to just **2 numbers**, then visualizes where each digit "lives" in that tiny 2-number space.

Think of it like this: imagine describing any human face using only 2 numbers. The network has to figure out the best 2 numbers to use so that similar faces get similar numbers.

---

## Part 1 - The Encoder

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**What it does:** Takes a big input (784 numbers) and squezes it down to a small latent code (2 numbers).

**Concepts covered:**

- `nn.Module` is PyTorch's base class for all neural networks. Every custom network inherits from it to get free functionality like parameter tracking and GPU support.

- `nn.Linear(input_dim, 128)` is a **fully connected layer** (every input neuron connects to every output neuron). The layer holds a weight matrix of shape (128 x 784) and learns which combinations of pixel values are meaningful.

- `torch.relu` is the **activation function**. Without it, stacking linear layers is mathematically equivalent to just one linear layer. ReLU (Rectified Linear Unit) simply does max(0, x) (it kills negative values and passes positive ones through). This introduces non-linearity, letting the network learn complex patterns.

- `super().__init__()` call the parent class constructor (required boilerplate in PyTorch). Skip it and things silently break.

- `forward()` defines the **computation graph** (what happens when data flows through). PyTorch traces this to compute gradients automatically.

The shape progression: 784 -> 128 -> 2. This is the compression happening step by step.

---

## Part 2 - The Decoder

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```

**What it does:** The mirror image of the encoder. Takes 2 numbers and tries to reconstruct the original 784-number image.

**Concepts covered:**

- The **decoder is symmetric** to the encoder: 2 -> 128 -> 784. This symmetry is a deliberate design choice (it forces the network to learn a two-way compression that's reversible).

- `torch.sigmoid` at the output is a crucial detail. Sigmoid squshes any number into the range (0, 1). Since pixel values are also normalized to (0, 1) by `transforms.ToTensor()`, this ensures the reconstructed pixels live in the same range as the original. If you used ReLU here, you could get values above 1, which wouldn't make sense as pixel intensities.

---

## Part 3 - The Autoencoder

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
```

**What it does:** Glues encoder and decoder into one network. Input goes in, reconstruction comes out.

**Concepts covered:**

- `z` is the **latent code** (the 2-number compressed representation). This is the thing you'll eventually visualize. The variable name `z` is conventional in ML literature for latent variables.

- The full pipelien is: image -> encoder -> z (2 numbers) -> decoder -> reconstructed image. The network is trained so that reconstruction matches the original as closely as possible.

- **Composing modules:** PyTorch lets you nest `nn.Module` objects inside each other. The autoencoder owns the encoder and decoder, and all their parameters are automatically tracked together.

---

## Part 4 - Training Setup

```python
input_dim = 784
latent_dim = 2
autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
```

**Concepts covered:**

- `784` comes from 28x28 pixels per MNIST image, all flattened into a single vector.

- `latent_dim = 2` is a bold choice (2 is the minimum needed to make a 2D scatter plot). In practice you'd use 32-256 for better reconstruction, but 2 makes visualization trivial and direct.

- `nn.MSELoss()` is **Mean Squared Error**, for each pixel, it computes (predicted - actual)^2, then averages across all pixels and all images. This is the signal that tells the network how badly it's doing, Lower MSE = better reconstruction.

- `optim.Adam` is an adaptive gradient descent optimizer. It adjusts the learning rate per parameter automatically, making it much more robust than plain gradient descent.

- `lr=0.001` is the global learning rate (a very standard starting value).

- `autoencoder.parameters()` passes all learnable weights to the optimizer so it knows what to update.

---

## Part 5 - Loading data

```python
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

**Concept covered:**

- `transforms.ToTensor()` does two things simultaneously: converts a PIL image to a PyTorch tensor, and rescales pixel values from integers (0-255) to floats (0.0-1.0). This normalization is essential (neural networks train poorly on large-magnitude inputs).

- `DataLoader` is PyTorch's data pipeline. It handles batching, shuffling, and parallel loading.

- `batch_size=64` means the network sees 64 images at once before updating its weights.\

- `shuffle=True` randomizes the order each epoch so the network doesn't memorize the sequence.

---

## The Big Problem With This Code

Here is the most important thing to notice: **the training loop is missing**

The code defines the network, the loss, and the optimizer but never actually calls them. It jumps straight to inference on an untrained network. What I am visualizing is a **random, untrained** latent space. The encoder is just randomly initialized weights.

A complete training loop should go between the optimizer line and the visualization:

```python
# THIS BLOCK IS MISSING FROM THE BLOG CODE
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for images, _ in trainloader:
        images = images.view(-1, 784)         # flatten 28x28 → 784
        optimizer.zero_grad()                  # clear old gradients
        reconstructed = autoencoder(images)    # forward pass
        loss = criterion(reconstructed, images) # compare output to input
        loss.backward()                        # compute gradients
        optimizer.step()                       # update weights
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(trainloader):.4f}")
```

Without this, the scatter plot will show 10 overlapping blob with no structure (the network hasn't learned anything yet).

---

## What the Visualization Shows (When Trained Correctly)

```python
plt.scatter(
    latent_representations[:, 0],
    latent_representations[:, 1],
    c=labels, cmap="viridis"
)
```

After proper training, each of the 60,000 images becomes a single dot in a 2D plot. The color encodes which digit(0-9) it is. A well-trained autoencoder will show these forming **distinct colored regions** (the network has learned to map each digit to a different region of the 2D space purely from pixel reconstruction pressure, without ever being told which digit is which).
