import os
import time
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .baseModel import BaseModel
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, channels, num_classes, latent_dim, embed_dim, conv_structure):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        layers = []
        in_channels = latent_dim + embed_dim

        if conv_structure:
            out_channels = conv_structure[0]
            layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=4, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(True))
            in_channels = out_channels
            
            for out_channels in conv_structure[1:]:
                layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=4, stride=2, padding=1, bias=False))
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU(True))
                in_channels = out_channels
            
            layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=channels,
                                             kernel_size=4, stride=2, padding=1, bias=False))
        else:
            raise ValueError("conv_structure for Generator must be provided.")
            
        self.main_module = nn.Sequential(*layers)
        self.output = nn.Tanh()

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels).view(noise.size(0), self.embed_dim, 1)
        x = torch.cat([noise, label_emb], dim=1)
        x = self.main_module(x)
        return self.output(x)


class Discriminator(nn.Module):
    def __init__(self, channels, num_classes, latent_feature_dim, conv_structure):
        super(Discriminator, self).__init__()

        layers = []
        in_channels = channels
        
        if conv_structure:
            for out_channels in conv_structure:
                layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=4, stride=2, padding=1, bias=False))
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                in_channels = out_channels
        else:
            raise ValueError("conv_structure for Discriminator must be provided.")
            
        self.main_module = nn.Sequential(*layers)
        self.out_layer = nn.Conv1d(in_channels=in_channels, out_channels=1,
                                   kernel_size=4, stride=1, padding=0, bias=False)
        self.label_embedding = nn.Embedding(num_classes, in_channels)

    def forward(self, x, labels):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        h = self.main_module(x)
        out = self.out_layer(h)

        out = torch.mean(out, dim=2)

        h_pool = torch.sum(h, dim=2)
        projection = torch.sum(self.label_embedding(labels) * h_pool, dim=1, keepdim=True)
        return out + projection


class WGAN_CP(BaseModel):
    def __init__(self, args, features, labels):
        super(WGAN_CP, self).__init__(args)
        
        self.num_classes = self.args.get("num_classes", 4)
        self.channels = self.args.get("channels", 1)
        self.generator_iters = self.args.get("generator_iters", 100)
        self.critic_iter = self.args.get("critic_iter", 5)
        self.batch_size = self.args.get("batch_size", 64)
        self.latent_dim = self.args.get("latent_dim", 100)
        self.learning_rate = self.args.get("learning_rate", 0.00005)
        self.weight_cliping_limit = self.args.get("weight_cliping_limit", 0.01)
        self.num_epoch = self.args.get("num_epoch", 100)
        
        generator_conv = self.args.get("generator_conv", [1024, 512, 256])
        discriminator_conv = self.args.get("discriminator_conv", 
                                           [256, 512, self.args.get("latent_feature_dim", 1024)])
        
        self.G = Generator(self.channels, self.num_classes, latent_dim=self.latent_dim, 
                           embed_dim=self.latent_dim, conv_structure=generator_conv)
        self.D = Discriminator(self.channels, self.num_classes, 
                               latent_feature_dim=self.args.get("latent_feature_dim", 1024),
                               conv_structure=discriminator_conv)
        
        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(features, dtype=torch.float32).unsqueeze(1),
                torch.tensor(labels, dtype=torch.long)
            ),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.G.to(self.device)
        self.D.to(self.device)

        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=self.learning_rate)
        scheduler_step = self.args.get("scheduler_step", 1000)
        scheduler_gamma = self.args.get("scheduler_gamma", 0.99)
        self.scheduler_d = torch.optim.lr_scheduler.StepLR(self.d_optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        self.scheduler_g = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    def to_var(self, x):
        return Variable(x).to(self.device)

    def discriminator_train_step(self, real_images, real_labels):
        self.D.train()

        for p in self.D.parameters():
            p.requires_grad = True
        
        one = torch.tensor(1.0, device=self.device)
        mone = -one

        self.D.zero_grad()

        for p in self.D.parameters():
            p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
        
        d_loss_real = self.D(real_images, real_labels).mean()
        d_loss_real.backward(one)

        z = torch.randn(self.batch_size, self.latent_dim, 1, device=self.device)
        fake_labels = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)
        fake_images = self.G(z, fake_labels)
        d_loss_fake = self.D(fake_images, fake_labels).mean()
        d_loss_fake.backward(mone)
        
        self.d_optimizer.step()
        return (d_loss_fake - d_loss_real).item()

    def generator_train_step(self):
        for p in self.D.parameters():
            p.requires_grad = False
        
        self.G.zero_grad()
        one = torch.tensor(1.0, device=self.device)
        
        z = torch.randn(self.batch_size, self.latent_dim, 1, device=self.device)
        fake_labels = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)
        fake_images = self.G(z, fake_labels)
        g_loss = self.D(fake_images, fake_labels).mean()
        g_loss.backward(one)
        self.g_optimizer.step()
        
        for p in self.D.parameters():
            p.requires_grad = True
        return -g_loss.item()

    def train(self):
        generator_history, discriminator_history = [], []
        
        early_stop_patience = self.args.get("early_stop_patience", None)
        if early_stop_patience is not None:
            best_gen_loss = float("inf")
            patience_counter = 0

        progress_bar = tqdm(range(self.num_epoch), desc="Training")
        for epoch in progress_bar:
            generator_losses = []
            discriminator_losses = []
            
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epoch}", leave=False):
                real_images = images.to(self.device)
                real_labels = labels.to(self.device)
                
                d_loss_batch = 0
                for _ in range(self.critic_iter):
                    d_loss_batch += self.discriminator_train_step(real_images, real_labels)
                d_loss_batch /= self.critic_iter
                
                g_loss = self.generator_train_step()
                
                discriminator_losses.append(d_loss_batch)
                generator_losses.append(g_loss)
            
            self.scheduler_d.step()
            self.scheduler_g.step()
            
            epoch_gen_loss = torch.tensor(generator_losses).mean().item()
            epoch_disc_loss = torch.tensor(discriminator_losses).mean().item()
            postfix = {
                'Epoch': f"{epoch+1}/{self.num_epoch}",
                'G_loss': epoch_gen_loss,
                'D_loss': epoch_disc_loss,
                'lr': self.scheduler_g.get_last_lr()[0]
            }
            progress_bar.set_postfix(postfix)
            generator_history.append(epoch_gen_loss)
            discriminator_history.append(epoch_disc_loss)
            
            if early_stop_patience is not None:
                if epoch_gen_loss < best_gen_loss:
                    best_gen_loss = epoch_gen_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
        return generator_history, discriminator_history

    def generate_samples(self, n_examples, labels):
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_examples, self.latent_dim, 1, device=self.device)
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            generated_images = self.G(z, labels)
        self.G.train()
        return generated_images.cpu().detach().numpy()
        
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        file_name = f"model_torch_final_{time.strftime('%Y%m%d')}.pt"
        full_path = os.path.join(path, file_name)
        
        torch.save({
            'generator': self.G.state_dict(),
            'discriminator': self.D.state_dict()
        }, full_path)

        return full_path
    
    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        self.G.load_state_dict(checkpoint['generator'])
        self.D.load_state_dict(checkpoint['discriminator'])