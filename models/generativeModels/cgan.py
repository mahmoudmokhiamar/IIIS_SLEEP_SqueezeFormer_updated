import os
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from .baseModel import BaseModel
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_size, output_length, class_num):
        super().__init__()
        
        self.z_size = z_size
        self.class_num = class_num
        self.output_length = output_length
        self.generator_layer_size = generator_layer_size

        self.label_emb = nn.Embedding(class_num, class_num)

        n_layers = len(generator_layer_size)

        self.L0 = output_length // (2 ** n_layers)
        
        if self.L0 < 1:
            raise ValueError("Output length is too short for the given number of layers.")

        init_channels = generator_layer_size[0]
        self.fc = nn.Linear(z_size + class_num, init_channels * self.L0)

        layers = []
        in_channels = init_channels

        for out_channels in generator_layer_size[1:]:
            layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(True))
            in_channels = out_channels

        layers.append(nn.ConvTranspose1d(in_channels, 1, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())
        self.deconv = nn.Sequential(*layers)

    def forward(self, z, labels):
        z = z.view(-1, self.z_size)
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(-1, self.generator_layer_size[0], self.L0)
        out = self.deconv(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, input_length, class_num):
        super().__init__()

        self.input_length = input_length
        self.class_num = class_num
        self.discriminator_layer_size = discriminator_layer_size

        self.label_emb = nn.Embedding(class_num, input_length)

        layers = []
        in_channels = 2
        current_length = input_length
        for out_channels in discriminator_layer_size:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
            current_length = current_length // 2

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels * current_length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        label_embedding = self.label_emb(labels).unsqueeze(1)
        
        x = torch.cat([x, label_embedding], dim=1)
        
        out = self.conv(x)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.squeeze()


class CGAN(BaseModel):
    def __init__(self, args, features, labels):
        super(CGAN, self).__init__(args)
        
        self.z_size = self.args['z_size']
        self.num_epoch = self.args['num_epoch']
        self.class_num = self.args['class_num']
        self.batch_size = self.args['batch_size']
        self.input_size = self.args['input_size']
        self.learning_rate = self.args['learning_rate']
        self.lr_decay_step = self.args.get('lr_decay_step', 4)
        self.lr_decay_gamma = self.args.get('lr_decay_gamma', 0.1)

        self.dataloader = DataLoader(
            TensorDataset(
                torch.tensor(features, dtype=torch.float32).unsqueeze(1),
                torch.tensor(labels, dtype=torch.long)
            ),
            batch_size=self.batch_size,
            shuffle=True
        )

        self.generator = Generator(self.args['generator_layer_size'], self.z_size, self.input_size, self.class_num).to(self.device)
        self.discriminator = Discriminator(self.args['discriminator_layer_size'], self.input_size, self.class_num).to(self.device)
        
        self.criterion = nn.BCELoss() 
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        self.g_scheduler = StepLR(self.g_optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_gamma)
        self.d_scheduler = StepLR(self.d_optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_gamma)

    def generator_train_step(self):
        self.g_optimizer.zero_grad()
        z = torch.randn(self.batch_size, self.z_size).to(self.device)
        fake_labels = torch.randint(0, self.class_num, (self.batch_size,)).to(self.device)
        fake_images = self.generator(z, fake_labels)
        validity = self.discriminator(fake_images, fake_labels)
        g_loss = self.criterion(validity, torch.ones(self.batch_size).to(self.device))
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss.item()

    def discriminator_train_step(self, real_images, real_labels):
        self.d_optimizer.zero_grad()
        
        real_validity = self.discriminator(real_images, real_labels)
        real_target = torch.ones(real_validity.size(), device=self.device)
        real_loss = self.criterion(real_validity, real_target)
        
        z = torch.randn(real_images.size(0), self.z_size).to(self.device)
        fake_labels = torch.randint(0, self.class_num, (real_images.size(0),)).to(self.device)
        fake_images = self.generator(z, fake_labels)
        
        fake_validity = self.discriminator(fake_images.detach(), fake_labels)
        fake_target = torch.zeros(fake_validity.size(), device=self.device)
        fake_loss = self.criterion(fake_validity, fake_target)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()

    def train(self):
        generator_history, discriminator_history = [], []
        progress_bar = tqdm(range(self.num_epoch), desc="Training")
        for epoch in progress_bar:
            generator_losses = []
            discriminator_losses = []

            for images, labels in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epoch}"):
                real_images = images.to(self.device)
                real_labels = labels.to(self.device)
                
                d_loss = self.discriminator_train_step(real_images, real_labels)
                g_loss = self.generator_train_step()

                discriminator_losses.append(d_loss)
                generator_losses.append(g_loss)

            self.g_scheduler.step()
            self.d_scheduler.step()

            postfix = {
                'Epoch': f"{epoch+1}/{self.num_epoch}",
                'Generator_loss': torch.tensor(generator_losses).mean().item(),
                'Discriminator_loss': torch.tensor(discriminator_losses).mean().item(),
                'lr': self.g_scheduler.get_last_lr()[0]
            }

            generator_history.append(torch.tensor(generator_losses).mean().item())
            discriminator_history.append(torch.tensor(discriminator_losses).mean().item())

            progress_bar.set_postfix(postfix)
        
        return generator_history, discriminator_history

    def generate_samples(self, n_ex, label):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_ex, self.z_size).to(self.device)
            labels = torch.full((n_ex,), label, dtype=torch.long).to(self.device)
            samples = self.generator(z, labels)
        self.generator.train()
        return samples.squeeze().cpu().detach().numpy()

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        file_name = f"model_torch_final_{time.strftime('%Y%m%d')}.pt"
        path = os.path.join(path, file_name)
        
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)

        return path

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])