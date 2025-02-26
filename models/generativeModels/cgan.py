import os
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset


### WGAN's implementation.

class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_size, input_size, class_num):
        super().__init__()
        
        self.z_size = z_size
        self.input_size = input_size
        self.label_emb = nn.Embedding(class_num, class_num)
        #change the approach of the generator from FCN to CNN.
        layers = []
        input_dim = z_size + class_num
        for hidden_size in generator_layer_size:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, input_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, z, labels):
        z = z.view(-1, self.z_size)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(-1, self.input_size)

class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, input_size, class_num):
        super().__init__()
        
        self.label_emb = nn.Embedding(class_num, class_num)
        self.input_size = input_size

        layers = []
        input_dim = input_size + class_num
        for hidden_size in discriminator_layer_size:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, labels):
        x = x.view(-1, self.input_size)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()

class CGAN(object):
    def __init__(self, args, features, labels):
        self.device = torch.device(args.get('device', 'cpu'))
        print(f"Using device: {self.device}")
        self.z_size = args['z_size']
        self.num_epoch = args['num_epoch']
        self.class_num = args['class_num']
        self.batch_size = args['batch_size']
        self.input_size = args['input_size']
        self.learning_rate = args['learning_rate']
        self.lr_decay_step = args.get('lr_decay_step', 4)
        self.lr_decay_gamma = args.get('lr_decay_gamma', 0.1)

        self.dataloader = DataLoader(
            TensorDataset(
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long)
            ),
            batch_size=self.batch_size,
            shuffle=True
        )

        self.generator = Generator(args['generator_layer_size'], self.z_size, self.input_size, self.class_num).to(self.device)
        self.discriminator = Discriminator(args['discriminator_layer_size'], self.input_size, self.class_num).to(self.device)
        
        self.criterion = nn.BCELoss() #wasserstein gan loss. 
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
        return samples.cpu().detach().numpy()

    def save_model(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        file_name = f"model_torch_final_{time.strftime('%Y%m%d')}.pt"
        path = os.path.join(dir, file_name)
        
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