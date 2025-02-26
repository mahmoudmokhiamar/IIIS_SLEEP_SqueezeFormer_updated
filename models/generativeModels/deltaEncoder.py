import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm

class Encoder(nn.Module):
    def __init__(self, feature_dim=256, encoder_size=[8192], z_dim=16, dropout=0.5, dropout_input=0.0, leak=0.2):
        super(Encoder, self).__init__()
        self.first_linear = nn.Linear(feature_dim*2, encoder_size[0])

        linear = []
        for i in range(len(encoder_size) - 1):
            linear.append(nn.Linear(encoder_size[i], encoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            # linear.append(nn.Tanh())
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)
        self.final_linear = nn.Linear(encoder_size[-1], z_dim)
        self.dropout_input = nn.Dropout(dropout_input)
        self.tanh = nn.Tanh()

    def forward(self, features, reference_features):
        features = self.dropout_input(features)
        x = torch.cat([features, reference_features], 1)

        x = self.first_linear(x)
        x = self.linear(x)

        x = self.final_linear(x)

        # return self.tanh(x)
        return x

class Decoder(nn.Module):
    def __init__(self, feature_dim=256, decoder_size=[8192], z_dim=16, dropout=0.5, leak=0.2):
        super(Decoder, self).__init__()
        self.first_linear = nn.Linear(z_dim+feature_dim, decoder_size[0])

        linear = []
        for i in range(len(decoder_size) - 1):
            linear.append(nn.Linear(decoder_size[i], decoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            # linear.append(nn.Tanh())
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)

        self.final_linear = nn.Linear(decoder_size[-1], feature_dim)
        self.tanh = nn.Tanh()

    def forward(self, reference_features, code):
        x = torch.cat([reference_features, code], 1)

        x = self.first_linear(x)
        x = self.linear(x)

        x = self.final_linear(x)

        # return self.tanh(x)
        return x

class DeltaEncoder(object):
    def __init__(self, args, features, labels, features_test, labels_test, resume = ''):
        self.count_data = 0
        self.num_epoch = args['num_epoch']
        self.noise_size = args['noise_size']
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.batch_size = args['batch_size']
        self.drop_out_rate = args['drop_out_rate']
        self.drop_out_rate_input = args['drop_out_rate_input']
        self.learning_rate = args['learning_rate']
        self.decay_factor = 0.8
        # self.num_shots = args['num_shots']
        self.resume = resume

        self.features, self.labels = features, labels
        self.features_test, self.labels_test = features_test, labels_test

        self.features_dim = self.features.shape[1]
        self.reference_features = self.random_pairs(self.features, self.labels)

        self._create_model()

    def random_pairs(self,X, labels):
        Y = X.copy()
        for l in range(labels.shape[1]):
            inds = np.where(labels[:,l])[0]
            inds_pairs = np.random.permutation(inds)
            Y[inds,:] = X[inds_pairs,:]
        return Y

    def _create_model(self):
        self.encoder = Encoder(self.features_dim, self.encoder_size, self.noise_size, self.drop_out_rate, self.drop_out_rate_input)
        self.decoder = Decoder(self.features_dim, self.decoder_size, self.noise_size, self.drop_out_rate)

        self.encoder = self.encoder
        self.decoder = self.decoder

        if self.resume:
            self.load_model()
    
    def load_model(self):
        checkpoint = torch.load(self.resume)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

    def loss(self, features_batch, reference_features_batch):
        l1loss = nn.L1Loss()

        self.pred_noise = self.encoder(features_batch, reference_features_batch)
        self.pred_x = self.decoder(reference_features_batch, self.pred_noise)

        abs_diff = l1loss(features_batch, self.pred_x)

        w = torch.pow(abs_diff, 2)
        w = w / torch.norm(w)

        loss = w * abs_diff

        return loss

    def optimizer(self, encoder, decoder, lr):
        optimizer = torch.optim.Adam([{'params': encoder.parameters()},
                                      {'params': decoder.parameters()}], lr=lr)

        return optimizer

    def next_batch(self, start, end):
        if start == 0:
            # if self.num_shots:
            #     self.reference_features = self.random_pairs(self.features, self.labels)
            idx = np.r_[:self.features.shape[0]]
            random.shuffle(idx)
            self.features = self.features[idx]
            self.reference_features = self.reference_features[idx]
            self.labels = self.labels[idx]
        if end > self.features.shape[0]:
            end = self.features.shape[0]

        return torch.from_numpy(self.features[start:end]).float(), \
               torch.from_numpy(self.reference_features[start:end]).float(), \
               torch.from_numpy(self.labels[start:end]).float()

    def train(self, save=False):
        last_loss_epoch = None
        train_history, test_history = [], []

        optimizer = self.optimizer(self.encoder, self.decoder, lr=self.learning_rate)

        self.encoder.train()
        self.decoder.train()

        postfix = {'Epoch': f"0/{self.num_epoch}", 'Train_loss': 0.0, 'Test_loss': 0.0, 'lr': self.learning_rate}
        progress_bar = tqdm(range(self.num_epoch), postfix=postfix, desc="Training")
        for epoch in progress_bar:
            mean_loss_e = 0.0
            
            for count in tqdm(range(0, self.features.shape[0], self.batch_size), desc=f'Epoch: {epoch+1}', leave=False):
                features_batch, reference_features_batch, labels_batch = self.next_batch(count, count + self.batch_size)

                with torch.enable_grad():
                    optimizer.zero_grad()
                    loss_e = self.loss(features_batch, reference_features_batch)
                    loss_e.backward()
                    optimizer.step()

                mean_loss_e += loss_e

                c = (count/self.batch_size) + 1

            mean_loss_e /= (self.features.shape[0] / self.batch_size)
            train_history.append(mean_loss_e.item())

            if last_loss_epoch is not None and mean_loss_e > last_loss_epoch:
                self.learning_rate *= self.decay_factor
            last_loss_epoch = mean_loss_e

            with torch.no_grad():
                self.encoder.eval()
                self.decoder.eval()

                test_loss = self.loss(
                    torch.from_numpy(self.features_test).float(),
                    torch.from_numpy(self.random_pairs(self.features_test, self.labels_test)).float()
                )
                test_history.append(test_loss.item())
            
            postfix = {'Epoch': f"{epoch + 1}/{self.num_epoch}", 'Train_loss': mean_loss_e.item(), 'Test_loss': test_loss.item(), 'lr': self.learning_rate}
            progress_bar.set_postfix(postfix)
        
        if save:
            self.save_model(self.encoder, self.decoder, "./model_torch_final.pt")

        return train_history, test_history

    def generate_samples(self, n_ex, class_label):
        self.encoder.eval()
        self.decoder.eval()

        n_ex = int(n_ex)

        features = np.zeros((n_ex, self.features.shape[1]))
        labels = np.zeros((n_ex, self.labels.shape[1]))

        class_indices = np.where(self.labels.argmax(axis=1) == class_label)[0]
        if len(class_indices) == 0:
            raise ValueError("The specified class label does not exist in the dataset.")

        references = torch.from_numpy(self.random_pairs(self.features[class_indices, ...], self.labels[class_indices, ...])).float()
        
        noise = self.encoder(
            torch.Tensor(self.features[class_indices, ...]),
            references
        )

        features = self.decoder(references, noise).cpu().detach().numpy()
        
        sampled_indices = np.random.choice(features.shape[0], n_ex, replace=True)
        features = features[sampled_indices, ...]

        labels = np.tile(class_label, (n_ex, 1))
        
        return features

    def save_model(self, save_dir):
        if not os.path.exists(os.path.dirname(save_dir)):
            os.mkdir(os.path.dirname(save_dir))
            
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            }, save_dir)