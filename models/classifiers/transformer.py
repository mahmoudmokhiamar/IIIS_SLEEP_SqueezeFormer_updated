import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
from .baseModel import BaseModel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

class TransformerClassifier(BaseModel):
    def __init__(self, args, X_train, y_train, X_test, y_test):
        super(TransformerClassifier, self).__init__(args)
        
        self.lr_decay_step = self.args.get('lr_decay_step', 4)
        self.lr_decay_gamma = self.args.get('lr_decay_gamma', 0.1)

        self.train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        self.test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        self.train_loader = DataLoader(self.train_dataset, batch_size=args['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args['batch_size'], shuffle=False)

        input_dim = X_train.shape[1]
        num_classes = len(torch.unique(torch.tensor(y_train)))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=args['d_model'],
            nhead=args['nhead'],
            dim_feedforward=args['dim_feedforward'],
            dropout=args['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args['num_encoder_layers'])

        self.model = nn.Sequential(
            nn.Linear(input_dim, args['d_model']),
            nn.ReLU(),
            self.transformer_encoder,
            nn.Flatten(),
            nn.Linear(args['d_model'], num_classes)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args['lr'])
        self.scheduler = StepLR(self.optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_gamma)

    def train_model(self):
        train_history = []
        test_history = []

        progress_bar = tqdm(range(self.args['num_epochs']), desc="Training", leave=True)
        for epoch in progress_bar:
            self.model.train()
            train_loss, train_correct, total = 0, 0, 0

            for X_batch, y_batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args['num_epochs']}", leave=False):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.model.zero_grad()
                # self.optimizer.zero_grad()
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

            test_loss_epoch = self.evaluate(epoch)
            test_history.append(test_loss_epoch)

            progress_bar.set_postfix({
                'Epoch': epoch + 1,
                'Train_loss': train_loss_epoch,
                'Test_loss': test_loss_epoch,
                'lr': self.scheduler.get_last_lr()[0]
            })

        return train_history, test_history

    def evaluate(self, epoch):
        self.model.eval()
        test_loss, test_correct, total = 0, 0, 0

        with torch.no_grad():
            test_progress = tqdm(self.test_loader, desc="Testing", postfix="Test", leave=False)
            for X_batch, y_batch in test_progress:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                test_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                test_correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

                test_progress.set_postfix({
                    'test_loss': test_loss / total,
                    'test_acc': test_correct / total
                })

        test_loss_epoch = test_loss / len(self.test_loader.dataset)
        return test_loss_epoch

    def predict(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()

    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))

        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)