import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import dataset
from scipy.stats import entropy

noise_size = 128

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3):
        super(UpscaleBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, self.padding, output_padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),

            # nn.Conv2d(out_channels, out_channels, kernel_size, 1, self.padding),
            # nn.BatchNorm2d(out_channels),
            # nn.PReLU()
        )
    
    def forward(self, x):
        x = self.block(x)
        return x

class ReduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3):
        super(ReduceBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),

            # nn.Conv2d(out_channels, out_channels, kernel_size, 1, self.padding),
            # nn.BatchNorm2d(out_channels),
            # nn.PReLU()
        )
    
    def forward(self, x):
        x = self.block(x)
        return x

class Generator(nn.Module):
    def __init__(self, noise_size=128):
        super(Generator, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(noise_size, 128*4*4),
            nn.Unflatten(dim=1, unflattened_size=(128, 4, 4)),

            UpscaleBlock(128, 64, 2, 3), # 128x4x4 => 64x8x8
            UpscaleBlock(64, 32, 2, 3), # 64x8x8 => 32x16x16
            UpscaleBlock(32, 16, 2, 3), # 32x16x16 => 16x32x32
            UpscaleBlock(16, 8, 2, 3), # 16x32x32 => 8x64x64

            nn.Conv2d(8, 3, 7, padding=3),
            nn.BatchNorm2d(3),
            nn.PReLU(),

            nn.Conv2d(3, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            ReduceBlock(3, 4, 2, 3), # 3x64x64 => 8x32x32
            nn.Dropout(0.1),
            ReduceBlock(4, 8, 2, 3), # 3x64x64 => 8x32x32
            nn.Dropout(0.1),
            ReduceBlock(8, 16, 2, 3), # 16x16x16 => 32x8x8
            nn.Dropout(0.1),
            ReduceBlock(16, 32, 2, 3), # 32x8x8 => 64x4x4

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(32*4*4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.block(x)
        return x

all_G_losses = []
all_D_losses = []
all_KLs = []

class MyDataset(Dataset):
    def __init__(self, all_dogs):
        from torchvision import transforms
        self.all_dogs = torch.tensor(all_dogs, dtype=torch.float32).permute(0, 3, 1, 2)
        print("Dataset shape:", self.all_dogs.shape, type(self.all_dogs))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.all_dogs)

    def __getitem__(self, idx):
        # img_transformed = self.transform(self.all_dogs[idx])
        return self.all_dogs[idx]

from scipy.stats import entropy

def compute_kl_divergence(batch, generated):
    batch_flat = batch.detach().numpy().flatten()
    generated_flat = generated.detach().numpy().flatten()
    # Гистограммы
    hist_batch, _ = np.histogram(batch_flat, bins=256, range=(0, 1), density=True)
    hist_generated, _ = np.histogram(generated_flat, bins=256, range=(0, 1), density=True)
    # Избегаем деления на ноль
    hist_generated = np.where(hist_generated == 0, 1e-10, hist_generated)
    return entropy(hist_batch, hist_generated)

'''
def one_epoch(model_D, model_G, dogs_dataloader, optimizer_D, optimizer_G, noise_size, g_loss, d_loss):
    # Teach D
    model_D.train()
    D_loss = 0
    for i, batch in enumerate(dogs_dataloader):
        optimizer_D.zero_grad()
        bs = len(batch)
        noise = torch.randn((bs, noise_size))
        generated = model_G(noise)
        predicted1 = model_D(generated)
        predicted2 = model_D(batch)
        loss = d_loss(predicted1, torch.zeros_like(predicted1))
        loss += d_loss(predicted2, torch.ones_like(predicted2))
        loss.backward()
        optimizer_D.step()
        D_loss += loss.item()
        if i % 20 == 0:
            print("D batch ", i, "/", len(dogs_dataloader))

    # Teach G
    G_loss = 0
    KL_divergence_total = 0
    model_G.train()
    for i, batch in enumerate(dogs_dataloader):
        optimizer_G.zero_grad()
        bs = len(batch)
        noise = torch.randn((bs, noise_size))
        generated = model_G(noise)
        predicted = model_D(generated)
        loss = g_loss(predicted, torch.ones_like(predicted))
        # kl_divergence = compute_kl_divergence(batch, generated)
        # loss += kl_divergence
        # KL_divergence_total += kl_divergence
        loss.backward()
        optimizer_G.step()
        G_loss += loss.item()
        if i % 20 == 0:
            print("G batch ", i, "/", len(dogs_dataloader))

    all_D_losses.append(D_loss / len(dogs_dataloader))
    all_G_losses.append(G_loss / len(dogs_dataloader))
    all_KLs.append(KL_divergence_total / len(dogs_dataloader))
    return D_loss / len(dogs_dataloader), G_loss / len(dogs_dataloader), KL_divergence_total / len(dogs_dataloader)
'''

def one_epoch(model_D, model_G, dogs_dataloader, optimizer_D, optimizer_G, noise_size, g_loss, d_loss):
    # Teach D
    D_loss = 0
    G_loss = 0
    KL_divergence_total = 0
    for i, batch in enumerate(dogs_dataloader):
        # D learning
        model_D.train()
        model_G.eval()
        optimizer_D.zero_grad()
        bs = len(batch)
        noise = torch.randn((bs, noise_size))
        generated = model_G(noise)
        predicted1 = model_D(generated)
        predicted2 = model_D(batch)
        dloss = d_loss(predicted1, torch.ones_like(predicted1) * 0.0)
        dloss += d_loss(predicted2, torch.ones_like(predicted2) * 1)
        dloss = dloss / 2
        dloss.backward()
        optimizer_D.step()
        D_loss += dloss.item()
        
        # G learning
        model_D.eval()
        model_G.train()
        optimizer_G.zero_grad()
        noise = torch.randn((bs, noise_size))
        generated = model_G(noise)
        predicted = model_D(generated)
        gloss = g_loss(predicted, torch.ones_like(predicted) * 1)
        gloss.backward()
        optimizer_G.step()
        if i % 20 == 0:
            print("Batch ", i+1, "/", len(dogs_dataloader))
            print("Predicted 1: ", predicted1)
            print("Predicted 2: ", predicted2)
        G_loss += gloss.item()

    all_D_losses.append(D_loss / len(dogs_dataloader))
    all_G_losses.append(G_loss / len(dogs_dataloader))
    all_KLs.append(KL_divergence_total / len(dogs_dataloader))
    return D_loss / len(dogs_dataloader), G_loss / len(dogs_dataloader), KL_divergence_total / len(dogs_dataloader)

def go_epochs(epochs, model_D, model_G, dogs_dataloader, optimizer_D, optimizer_G, noise_size, g_loss, d_loss):
    for epoch in range(epochs - 1):
        D_loss, G_loss, KL_div = one_epoch(model_D, model_G, dogs_dataloader, optimizer_D, optimizer_G, noise_size, g_loss, d_loss)
        print("Discriminator loss:", D_loss, "Generator loss:", G_loss, "KL:", KL_div)
    return one_epoch(model_D, model_G, dogs_dataloader, optimizer_D, optimizer_G, noise_size, g_loss, d_loss)

g_loss = nn.MSELoss()
d_loss = nn.BCELoss()
dogs_dataset = MyDataset(dataset.full_dataset)
dogs_dataloader = DataLoader(dogs_dataset, batch_size=32, shuffle=True)

generator = Generator(noise_size)
discriminator = Discriminator()
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0005)