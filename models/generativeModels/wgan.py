import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
from .baseModel import BaseModel
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10, embed_dim=50):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, embed_dim)
        self.init_size = 16

        self.l1 = nn.Sequential(nn.Linear(latent_dim + embed_dim, 256 * self.init_size))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(32, 16, 4, 2, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(16, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), dim=1)
        out = self.l1(gen_input)
        out = out.view(noise.size(0), 256, self.init_size)
        eeg = self.conv_blocks(out)
        return eeg

class Discriminator(nn.Module):
    def __init__(self, n_classes=10, embed_dim=128 * 32):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 32, 1)
        self.label_emb = nn.Embedding(n_classes, embed_dim)

    def forward(self, eeg, labels):
        out = self.conv_blocks(eeg)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        label_embedding = self.label_emb(labels)
        projection = torch.sum(out * label_embedding, dim=1, keepdim=True)
        return validity + projection

class WGANModel(BaseModel):
    def __init__(self, args, features, labels):
        super(WGANModel, self).__init__(args)
        self.n_epochs    = self.args.get('n_epochs', 200)
        self.batch_size  = self.args.get('batch_size', 64)
        self.lr          = self.args.get('lr', 0.00005)
        self.latent_dim  = self.args.get('latent_dim', 100)
        self.n_critic    = self.args.get('n_critic', 5)
        self.clip_value  = self.args.get('clip_value', 0.01)
        self.early_stopping_patience = self.args.get('early_stopping_patience', None)
        self.scheduler_step_size = self.args.get('scheduler_step_size', None)
        self.scheduler_gamma = self.args.get('scheduler_gamma', 0.1)
        
        self.n_classes = self.args.get('n_classes', 10)

        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.long)
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        self.dataset = TensorDataset(features, labels)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.generator = Generator(latent_dim=self.latent_dim, n_classes=self.n_classes).to(self.device)
        self.discriminator = Discriminator(n_classes=self.n_classes).to(self.device)

        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        if self.scheduler_step_size is not None:
            self.scheduler_G = optim.lr_scheduler.StepLR(
                self.optimizer_G,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma
            )
            self.scheduler_D = optim.lr_scheduler.StepLR(
                self.optimizer_D,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma
            )
        else:
            self.scheduler_G = None
            self.scheduler_D = None

        self.generator_history = []
        self.discriminator_history = []
        self.best_loss = float('inf')
        self.epochs_no_improve = 0

    def train(self):
        for epoch in range(self.n_epochs):
            epoch_G_loss = 0.0
            epoch_D_loss = 0.0
            last_g_loss = 0.0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.n_epochs}", leave=False)
            for i, (real_eeg, real_labels) in enumerate(pbar):
                real_eeg = real_eeg.to(self.device)
                real_labels = real_labels.to(self.device)
                
                self.optimizer_D.zero_grad()
                z = torch.randn(real_eeg.size(0), self.latent_dim, device=self.device)

                fake_eeg = self.generator(z, real_labels).detach()
                loss_D = -torch.mean(self.discriminator(real_eeg, real_labels)) + torch.mean(self.discriminator(fake_eeg, real_labels))
                loss_D.backward()
                self.optimizer_D.step()
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)
                epoch_D_loss += loss_D.item()

                if i % self.n_critic == 0:
                    self.optimizer_G.zero_grad()
                    gen_eeg = self.generator(z, real_labels)
                    loss_G = -torch.mean(self.discriminator(gen_eeg, real_labels))
                    loss_G.backward()
                    self.optimizer_G.step()
                    epoch_G_loss += loss_G.item()
                    last_g_loss = loss_G.item()

                pbar.set_postfix({
                    'D_loss': f"{loss_D.item():.4f}",
                    'G_loss': f"{last_g_loss:.4f}"
                })

            avg_G_loss = epoch_G_loss / (len(self.dataloader) / self.n_critic) if len(self.dataloader) >= self.n_critic else epoch_G_loss
            avg_D_loss = epoch_D_loss / len(self.dataloader)
            self.generator_history.append(avg_G_loss)
            self.discriminator_history.append(avg_D_loss)

            if self.scheduler_G is not None:
                self.scheduler_G.step()
            if self.scheduler_D is not None:
                self.scheduler_D.step()

            if self.early_stopping_patience is not None:
                if avg_G_loss < self.best_loss:
                    self.best_loss = avg_G_loss
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break

        return self.generator_history, self.discriminator_history

    def generate_samples(self, n_ex, label=None):
        z = torch.randn(n_ex, self.latent_dim, device=self.device)
        if label is None:
            label = torch.randint(0, self.n_classes, (n_ex,), device=self.device)
        else:   
            if not isinstance(label, torch.Tensor):
                label = torch.full((n_ex,), label, dtype=torch.long, device=self.device)
            else:
                label = label.to(self.device)
        gen_eeg = self.generator(z, label)
        return gen_eeg.squeeze().detach().cpu().numpy()

    def save_model(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, path)
        return path

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])