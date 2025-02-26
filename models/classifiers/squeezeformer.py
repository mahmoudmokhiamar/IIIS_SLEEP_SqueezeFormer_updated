import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
from .baseModel import BaseModel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeFormerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, conv_kernel_size=3, dropout=0.1, ffn_expansion_factor=4):
        super(SqueezeFormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion_factor, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                 num_heads=n_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size,
                      padding=conv_kernel_size // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion_factor, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.ffn1(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        attn_output, _ = self.self_attn(x, x, x)
        x = attn_output + residual

        residual = x
        x = self.norm3(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv_module(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = x_conv + residual

        residual = x
        x = self.norm4(x)
        x = self.ffn2(x)
        x = x + residual

        return x

class SqueezeFormer(nn.Module):
    def __init__(self, input_dim, num_classes, model_size='small', num_layers=None, dropout=0.1):
        super(SqueezeFormer, self).__init__()
        if model_size == 'small':
            d_model = 64
            n_heads = 4
            conv_kernel_size = 3
            ffn_expansion_factor = 4
            num_layers = 4 if num_layers is None else num_layers
        elif model_size == 'medium':
            d_model = 128
            n_heads = 8
            conv_kernel_size = 3
            ffn_expansion_factor = 4
            num_layers = 6 if num_layers is None else num_layers
        elif model_size == 'large':
            d_model = 256
            n_heads = 8
            conv_kernel_size = 3
            ffn_expansion_factor = 4
            num_layers = 8 if num_layers is None else num_layers
        else:
            raise ValueError("model_size must be 'small', 'medium', or 'large'.")

        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        self.layers = nn.ModuleList([
            SqueezeFormerBlock(d_model, n_heads, conv_kernel_size, dropout, ffn_expansion_factor)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)

        if seq_len > self.pos_embedding.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum supported length {self.pos_embedding.size(1)}")
        x = x + self.pos_embedding[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)

        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        logits = self.classifier(x)
        return logits

class SqueezeFormerClassifier(BaseModel):
    def __init__(self, args, X_train, y_train, X_test, y_test):
        super(SqueezeFormerClassifier, self).__init__(args)

        self.lr_decay_step = self.args.get('lr_decay_step', 4)
        self.lr_decay_gamma = self.args.get('lr_decay_gamma', 0.1)

        self.train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long)
        )
        self.test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32), 
            torch.tensor(y_test, dtype=torch.long)
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=args['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args['batch_size'], shuffle=False)

        input_dim = X_train.shape[2]
        num_classes = len(torch.unique(torch.tensor(y_train)))

        self.model = SqueezeFormer(
            input_dim=input_dim,
            num_classes=num_classes,
            model_size=args.get('model_size', 'small')
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args['lr'])
        self.scheduler = StepLR(self.optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_gamma)

    def train_model(self):
        train_history = []
        test_history = []

        patience = max(1, int(0.1 * self.args['num_epochs']))
        min_delta = self.args.get('min_delta', 0.05)

        best_loss = float('inf')
        epochs_without_improvement = 0

        progress_bar = tqdm(range(self.args['num_epochs']), desc="Training", leave=True)
        for epoch in progress_bar:
            self.model.train()
            train_loss, train_correct, total = 0, 0, 0

            for X_batch, y_batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args['num_epochs']}", leave=False):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                train_correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
            
            self.scheduler.step()

            train_loss_epoch = train_loss / len(self.train_loader.dataset)
            train_history.append(train_loss_epoch)

            test_loss_epoch = self.evaluate()
            test_history.append(test_loss_epoch)

            if best_loss - test_loss_epoch > min_delta:
                best_loss = test_loss_epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            progress_bar.set_postfix({
                'Epoch': epoch + 1,
                'Train_loss': train_loss_epoch,
                'Test_loss': test_loss_epoch,
                'lr': self.scheduler.get_last_lr()[0],
                'NoImprovement': epochs_without_improvement
            })

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}: "
                      f"no improvement of at least {min_delta} for {patience} consecutive epochs.")
                break

        return train_history, test_history

    def evaluate(self):
        self.model.eval()
        test_loss, test_correct, total = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                test_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                test_correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        test_loss_epoch = test_loss / len(self.test_loader.dataset)
        return test_loss_epoch
    
    def predict(self, X):
        self.model.eval()
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)

        if X.ndim == 2:
            X = X.unsqueeze(0)

        X = X.to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            _, predictions = outputs.max(1)
        return predictions.cpu().numpy()
        
    def save_model(self, file_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args
        }
        torch.save(checkpoint, file_path)
        print(f"Model checkpoint saved to: {file_path}")

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file '{file_path}' not found.")

        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.args = checkpoint.get('args', self.args)
        print(f"Model checkpoint loaded from: {file_path}")