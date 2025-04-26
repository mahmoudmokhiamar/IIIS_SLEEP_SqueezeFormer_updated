import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .baseModel import BaseModel
from torch.utils.data import DataLoader, TensorDataset

class EEGNetSingleChannel(BaseModel):
    def __init__(self, args, X_train, y_train, X_test, y_test):
        super(EEGNetSingleChannel, self).__init__(args)
        
        F1 = self.args.get('F1', 16)
        D = self.args.get('D', 4)
        Samples = self.args.get('Samples', 512)
        kernLength = self.args.get('kernLength', 10)
        nb_classes = self.args.get('nb_classes', 4)
        dropoutRate = self.args.get('dropoutRate', 0.5)

        F2 = F1 * D

        self.conv_temporal = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm_temporal = nn.BatchNorm2d(F1)

        self.conv_depthwise = nn.Conv2d(F1, F1 * D, (1, 1), groups=F1, bias=False)
        self.batchnorm_depthwise = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout2d(dropoutRate)

        self.conv_separable = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm_separable = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout2d(dropoutRate)

        reduced_time = Samples // (4 * 8)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * reduced_time, nb_classes)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)

        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        batch_size = self.args.get('batch_size', 32)
        self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train),
                                       batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(self.X_test, self.y_test),
                                      batch_size=batch_size, shuffle=False)

        self.epochs = self.args.get('epochs', 10)
        self.lr = self.args.get('learning_rate', 1e-3)
        
        self.to(self.device)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(1)

        x = self.conv_temporal(x)
        x = self.batchnorm_temporal(x)
        x = self.conv_depthwise(x)
        x = self.batchnorm_depthwise(x)
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.conv_separable(x)
        x = self.batchnorm_separable(x)
        x = self.activation2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
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
            accuracy = 100 * correct / total

            progress_bar.set_postfix({
                "train_loss": float(avg_train_loss),
                "test_loss": float(avg_test_loss),
                "test_accuracy": float(accuracy)
            })

            train_history.append(avg_train_loss)
            test_history.append(avg_test_loss)

        return train_history, test_history

    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.unsqueeze(1)
        
        x = x.to(self.device)

        self.eval()
        with torch.no_grad():
            output = self(x)
            _, predicted = torch.max(output.data, 1)

        return predicted.cpu().detach().numpy()


    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        print(f'Model loaded from {path}')