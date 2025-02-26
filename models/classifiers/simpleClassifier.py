import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .baseModel import BaseModel
from torch.utils.data import DataLoader, TensorDataset

class SimpleClassifier(BaseModel):
    def __init__(self, args, X_train, y_train, X_test, y_test):
        super(SimpleClassifier, self).__init__(args)

        self.input_dim = self.args['input_dim']
        self.output_dim = self.args['output_dim']
        self.epochs = self.args.get('epochs', 10)
        self.lr = self.args.get('lr', 0.001)
        self.device = torch.device(self.args.get('device', 'cpu'))
        print(f"Using device: {self.device}")
        self.hidden_dims = self.args.get('hidden_dims', [256, 128, 64])

        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()),
            batch_size=self.args.get('batch_size', 64),
            shuffle=True
        )

        self.test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()),
            batch_size=args.get('batch_size', 64),
            shuffle=False
        )


        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        x = self.softmax(x)
        return x

    def train_model(self):
        train_history, test_history = [], []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        progress_bar = tqdm(range(self.epochs), desc="Training", leave=True)
        for epoch in progress_bar:
            self.train()
            total_train_loss = 0
            for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):
                # print(f"Type of x: {type(x)}, Type of y: {type(y)}")
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                if y.dim() > 1:
                    y = torch.argmax(y, dim=1)

                output = self(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_loader)

            self.eval()
            total_test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)

                    if y.dim() > 1:
                        y = torch.argmax(y, dim=1)

                    output = self(x)
                    loss = criterion(output, y)
                    total_test_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            avg_test_loss = total_test_loss / len(self.test_loader)
            accuracy = 100 * correct / total #scalar conversion

            # Corrected: No need to call .cpu() or .numpy()
            progress_bar.set_postfix({
                "train_loss": float(avg_train_loss), #dealing with scalar
                "test_loss": float(avg_test_loss),
                "test_accuracy":float( accuracy)
            })

            train_history.append(avg_train_loss)
            test_history.append(avg_test_loss)

        return train_history, test_history

    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).float()
        x = x.to(self.device)

        self.eval()
        with torch.no_grad():
            output = self(x)
            _, predicted = torch.max(output.data, 1)
        return predicted.cpu().detach().numpy()
        
    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))

        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))