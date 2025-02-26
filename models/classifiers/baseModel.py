import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.device = torch.device(self.args.get('device', 'cpu'))
        print(f"Using device: {self.device}")

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass
    
    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass