"""
Healthcare GAN Implementation for Synthetic Data Generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class HealthcareGAN:
    def __init__(self, input_dim, hidden_dim, output_dim, device='cuda'):
        self.device = device
        self.generator = Generator(input_dim, hidden_dim, output_dim).to(device)
        self.discriminator = Discriminator(output_dim, hidden_dim).to(device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train_step(self, real_data):
        batch_size = real_data.size(0)
        real_label = torch.ones(batch_size, 1).to(self.device)
        fake_label = torch.zeros(batch_size, 1).to(self.device)

        # Train Discriminator
        self.d_optimizer.zero_grad()
        label = ((real_label + fake_label) / 2) + torch.rand(batch_size, 1).to(self.device) * 0.1
        
        d_output_real = self.discriminator(real_data)
        d_loss_real = self.criterion(d_output_real, real_label)
        
        noise = torch.randn(batch_size, self.generator.net[0].in_features).to(self.device)
        fake_data = self.generator(noise)
        d_output_fake = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(d_output_fake, fake_label)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        g_output = self.discriminator(fake_data)
        g_loss = self.criterion(g_output, real_label)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_x': d_output_real.mean().item(),
            'd_g_z': g_output.mean().item()
        }

    def generate_samples(self, num_samples):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.generator.net[0].in_features).to(self.device)
            samples = self.generator(noise)
        self.generator.train()
        return samples.cpu().numpy()

    def train(self, dataloader, num_epochs, save_path=None):
        for epoch in range(num_epochs):
            g_losses, d_losses = [], []
            for batch_data in dataloader:
                batch_data = batch_data.to(self.device)
                metrics = self.train_step(batch_data)
                g_losses.append(metrics['g_loss'])
                d_losses.append(metrics['d_loss'])
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"G_loss: {np.mean(g_losses):.4f} "
                  f"D_loss: {np.mean(d_losses):.4f}")

            if save_path and (epoch + 1) % 10 == 0:
                torch.save({
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                }, f"{save_path}/gan_checkpoint_epoch_{epoch+1}.pt") 