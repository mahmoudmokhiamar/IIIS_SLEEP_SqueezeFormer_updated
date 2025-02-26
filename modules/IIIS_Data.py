import os
import json
import time
import torch
import logging
import warnings
import numpy as np
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
from collections import Counter
from keras.utils import to_categorical

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from models.generativeModels import CGAN, DeltaEncoder
from models.classifiers import SimpleClassifier, TransformerClassifier, EEGNetSingleChannel, SqueezeFormerClassifier

warnings.filterwarnings("ignore")

class IIIS_DATA(object):

    def __init__(self, config):
        self.config = config
        self.experiment_name = f'{self.config["generative_model"]}_{self.config["classifier_model"]}_{time.strftime("%Y%m%d_%H%M%S")}'
        self.experiment_name = self.config.get('experiment_name', self.experiment_name)

        self.experiment_name = os.path.join('./experiments', self.experiment_name)
        os.makedirs(self.experiment_name, exist_ok=True)
        
        log_file = os.path.join(self.experiment_name, "experiment.log")

        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file, mode="w")
        file_formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter("%(asctime)s - %(message)s")
        stream_handler.setFormatter(stream_formatter)
        self.logger.addHandler(stream_handler)

        self._load_data(
            path=self.config['data_path'],
            test_size=self.config.get('test_size', 0.2),
            standardize=self.config.get('standardize', True),
            stratify=self.config.get('stratify', True)
        )

        generative_model = self.config.get('generative_model', 'CGAN').lower()
        if 'generative_model_args' not in self.config:
            raise ValueError('generative_model_args not found in config')
        generative_model_args = self.config['generative_model_args']

        if generative_model == 'cgan':
            self.generator = CGAN(generative_model_args, self.X, self.y)
            self.logger.info('Generator set to CGAN')
        elif generative_model == 'deltaencoder':
            X_train_under, y_train_under = self._random_undersample()
            self.generator = DeltaEncoder(
                generative_model_args, X_train_under, to_categorical(y_train_under), 
                self.X_test, to_categorical(self.y_test)
            )
            self.logger.info('Generator set to DeltaEncoder')

        classifier_model = self.config.get('classifier_model', 'SimpleClassifier').lower()
        with open(os.path.join(self.experiment_name, f"{generative_model}_{classifier_model}_params.json"), "w") as file:
            json.dump(self.config, file, indent=4)

    def process(self):
        k_folds = self.config.get('k_folds', 3)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

        results_before_balance = []
        results_after_balance = []

        if self.generator:
            self.logger.info("Training Generative Model...")
            generator_history, discriminator_history = self.generator.train()

            if self.config.get('generative_model', 'CGAN').lower() == 'cgan':
                self.logger.info(f"Generator Loss: {generator_history[-1]}")
                self.logger.info(f"Discriminator Loss: {discriminator_history[-1]}")
            else:
                self.logger.info(f"Train Loss: {generator_history[-1]}")
                self.logger.info(f"Test Loss: {discriminator_history[-1]}")
            
            os.makedirs(os.path.join(self.experiment_name, 'models/'), exist_ok=True)
            self.generator.save_model(os.path.join(self.experiment_name, 'models/', 'generator.pth'))

            self._plot_generator_discriminator_loss(generator_history, discriminator_history)
            self._plot_signal_visualization()
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            self.logger.info(f"Starting Fold {fold}/{k_folds}")

            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            self.logger.info(f"Training Classifier Before Balancing for Fold {fold}")
            classifier_before = self._initialize_classifier((X_train, y_train), (X_val, y_val))
            train_history_before, test_history_before = classifier_before.train_model()

            if type(classifier_before).__name__ == 'EEGNetSingleChannel':
                preds_before = classifier_before.predict(X_val.reshape(X_val.shape[0], 1, 1, X_val.shape[1]))
            elif type(classifier_before).__name__ == 'SqueezeFormerClassifier':
                preds_before = classifier_before.predict(X_val.reshape(X_val.shape[0], 1, X_val.shape[1]))
            else:
                preds_before = classifier_before.predict(X_val)     

            accuracy_before = accuracy_score(y_val, preds_before)
            report_before = classification_report(y_val, preds_before, output_dict=True)

            classifier_before.save_model(os.path.join(self.experiment_name, 'models/', f"classifier_fold_{fold}.pth"))

            self.logger.info(f"Fold {fold} Accuracy Before Balancing: {accuracy_before}")
            self.logger.info(f"Fold {fold} Report Before Balancing: \n{classification_report(y_val, preds_before)}")

            results_before_balance.append({
                "fold": fold,
                "accuracy": accuracy_before,
                "f1": report_before["weighted avg"]["f1-score"],
                "precision": report_before["weighted avg"]["precision"],
                "recall": report_before["weighted avg"]["recall"]
            })

            self._plot_loss_curves(train_history_before, test_history_before, f"fold_{fold}_before_balance")
            self._plot_confusion_matrix(y_val, preds_before, f"fold_{fold}_before_balance")

            if self.generator:
                self.logger.info('Balanced data generated.')
                
                X_train_balanced, y_train_balanced = self._balance_data(X_train, y_train)
                self._plot_class_distribution(y_train, y_train_balanced, prefix=f"fold_{fold}")
            else:
                X_train_balanced, y_train_balanced = X_train, y_train

            self.logger.info(f"Training Classifier After Balancing for Fold {fold}")
            classifier_after = self._initialize_classifier((X_train_balanced, y_train_balanced), (X_val, y_val))
            train_history_after, test_history_after = classifier_after.train_model()

            if type(classifier_before).__name__ == 'EEGNetSingleChannel':
                preds_after = classifier_after.predict(X_val.reshape(X_val.shape[0], 1, 1, X_val.shape[1]))
            elif type(classifier_before).__name__ == 'SqueezeFormerClassifier':
                preds_after = classifier_after.predict(X_val.reshape(X_val.shape[0], 1, X_val.shape[1]))
            else:
                preds_after = classifier_after.predict(X_val)
            
            accuracy_after = accuracy_score(y_val, preds_after)
            report_after = classification_report(y_val, preds_after, output_dict=True)

            classifier_after.save_model(os.path.join(self.experiment_name, 'models/', f"classifier_fold_{fold}_after_balance.pth"))

            self.logger.info(f"Fold {fold} Accuracy After Balancing: {accuracy_after}")
            self.logger.info(f"Fold {fold} Report After Balancing: \n{classification_report(y_val, preds_after)}")

            results_after_balance.append({
                "fold": fold,
                "accuracy": accuracy_after,
                "f1": report_after["weighted avg"]["f1-score"],
                "precision": report_after["weighted avg"]["precision"],
                "recall": report_after["weighted avg"]["recall"]
            })

            self._plot_loss_curves(train_history_after, test_history_after, f"fold_{fold}_after_balance")
            self._plot_confusion_matrix(y_val, preds_after, f"fold_{fold}_after_balance")

        metrics_before = {
            "accuracy": np.mean([result["accuracy"] for result in results_before_balance]),
            "f1": np.mean([result["f1"] for result in results_before_balance]),
            "precision": np.mean([result["precision"] for result in results_before_balance]),
            "recall": np.mean([result["recall"] for result in results_before_balance])
        }

        metrics_after = {
            "accuracy": np.mean([result["accuracy"] for result in results_after_balance]),
            "f1": np.mean([result["f1"] for result in results_after_balance]),
            "precision": np.mean([result["precision"] for result in results_after_balance]),
            "recall": np.mean([result["recall"] for result in results_after_balance])
        }

        self.logger.info(f"Average Metrics Before Balancing across {k_folds} folds: {metrics_before}")
        self.logger.info(f"Average Metrics After Balancing across {k_folds} folds: {metrics_after}")
        torch.cuda.empty_cache()
        return metrics_before, metrics_after
    
    def _initialize_classifier(self, train_data, val_data):
        X_train, y_train = train_data
        X_val, y_val = val_data

        classifier_model = self.config.get('classifier_model', 'SimpleClassifier').lower()
        classifier_args = self.config['classifier_args']

        if classifier_model == 'simpleclassifier':
            return SimpleClassifier(classifier_args, X_train, y_train, X_val, y_val)
        elif classifier_model == 'transformer':
            return TransformerClassifier(classifier_args, X_train, y_train, X_val, y_val)
        elif classifier_model == 'eegnet':            
            return EEGNetSingleChannel(
                classifier_args, 
                X_train.reshape(X_train.shape[0], 1, 1, X_train.shape[1]),
                y_train,
                X_val.reshape(X_val.shape[0], 1, 1, X_val.shape[1]),
                y_val
            )
        elif classifier_model == 'squeezeformer':
            return SqueezeFormerClassifier(
                classifier_args, 
                X_train.reshape(X_train.shape[0], 1, X_train.shape[1]),
                y_train,
                X_val.reshape(X_val.shape[0], 1, X_val.shape[1]),
                y_val
            )
        else:
            raise ValueError(f"Unknown classifier model: {classifier_model}")

    def _plot_loss_curves(self, train_losses, test_losses, title_suffix):
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title(f"Loss Curves - {title_suffix}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = os.path.join(self.experiment_name, f"loss_curves_{title_suffix.replace(' ', '_')}.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved loss curves plot to {plot_path}")

    def _plot_generator_discriminator_loss(self, generator_losses, discriminator_losses):
        if 'gan' in self.config.get('generative_model', 'CGAN').lower():
            plt.figure()
            plt.plot(generator_losses, label='Generator Loss')
            plt.plot(discriminator_losses, label='Discriminator Loss')
            plt.title("Generator and Discriminator Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plot_path = os.path.join(self.experiment_name, "generative_model_loss_curve.png")
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Saved generator/discriminator loss plot to {plot_path}")
        else:
            plt.figure()
            plt.plot(generator_losses, label='Train Loss')
            plt.plot(discriminator_losses, label='Test Loss')
            plt.title("Train and Test Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plot_path = os.path.join(self.experiment_name, "generative_model_loss_curve.png")
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Saved train/test loss plot to {plot_path}")

    def _plot_class_distribution(self, original_labels, balanced_labels, prefix=""):
        plt.figure()
        sns.countplot(x=original_labels)
        plt.title("Class Distribution Before Balancing")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        before_path = os.path.join(self.experiment_name, f"{prefix}_class_distribution_before.png")
        plt.savefig(before_path)
        plt.close()
        self.logger.info(f"Saved class distribution (before) plot to {before_path}")

        plt.figure()
        sns.countplot(x=balanced_labels)
        plt.title("Class Distribution After Balancing")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        after_path = os.path.join(self.experiment_name, f"{prefix}_class_distribution_after.png")
        plt.savefig(after_path)
        plt.close()
        self.logger.info(f"Saved class distribution (after) plot to {after_path}")

    def _plot_confusion_matrix(self, y_true, y_pred, title_suffix):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {title_suffix}")
        cm_path = os.path.join(self.experiment_name, f"confusion_matrix_{title_suffix.replace(' ', '_')}.png")
        plt.savefig(cm_path)
        plt.close()
        self.logger.info(f"Saved confusion matrix plot to {cm_path}")

    def _plot_signal_visualization(self):
        classes = np.unique(self.y_train)
        plt.figure(figsize=(25, 5 * len(classes)))

        for i, c in enumerate(classes):
            temp = to_categorical(self.y_test)
            mask = temp.argmax(axis=1) == c

            random_index = np.random.randint(0, self.X_test[mask].shape[0])
            original_sample = self.X_test[random_index]
            generated_sample = self.generator.generate_samples(1, c)[0]
            generated_sample = self.scaler.transform(generated_sample.reshape(1, -1))[0]

            plt.subplot(len(classes), 4, i * 4 + 1)
            plt.plot(generated_sample, label='Generated', color='green')
            plt.title(f'Class {c} - Generated')
            plt.legend()

            plt.subplot(len(classes), 4, i * 4 + 2)
            plt.plot(original_sample, label='Original', color='blue')
            plt.title(f'Class {c} - Original')
            plt.legend()

            if isinstance(self.generator, DeltaEncoder):
                references = torch.from_numpy(self.generator.random_pairs(self.X_test[mask][random_index].reshape(1, -1), temp[mask][random_index].reshape(1, -1))).float()
                noise = self.generator.encoder(
                    torch.tensor(self.X_test[mask][random_index].reshape(1, -1)).float(),
                    references
                )
                reconstructed_sample = self.generator.decoder(references, noise).cpu().detach().numpy()[0]
                reconstructed_sample = self.scaler.transform(reconstructed_sample.reshape(1, -1))[0]

                plt.subplot(len(classes), 4, i * 4 + 3)
                plt.plot(reconstructed_sample, label='Reconstructed', color='orange')
                plt.title(f'Class {c} - Reconstructed')
                plt.legend()

                plt.subplot(len(classes), 4, i * 4 + 4)
                plt.plot(original_sample, label='Original', color='blue')
                plt.plot(reconstructed_sample, label='Reconstructed', color='orange', linestyle='dotted')
                plt.plot(generated_sample, label='Generated', color='green', linestyle='dashed')
                plt.title('Delta Encoder Visualization')
                plt.legend()
            
            plt.subplot(len(classes), 4, i * 4 + 3)
            plt.plot(original_sample, label='Original', color='blue')
            plt.plot(generated_sample, label='Generated', color='green', linestyle='dashed')
            plt.title('Delta Encoder Visualization')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_name, 'signals_visualization_plot.png'))
        plt.close()

        self.logger.info(f"Saved Delta Encoder visualization plot to {self.experiment_name}")

    def _balance_data(self, X_train, y_train, target_ratio=1.0):
        class_counts = Counter(y_train)
        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]

        target_counts = {cls: int(majority_count * target_ratio) for cls in class_counts}

        X_balanced = []
        y_balanced = []

        for cls, count in class_counts.items():
            X_cls = X_train[y_train == cls]
            y_cls = y_train[y_train == cls]

            X_balanced.append(X_cls)
            y_balanced.append(y_cls)

            if count < target_counts[cls]:
                n_to_generate = target_counts[cls] - count
                synthetic_samples = self.generator.generate_samples(n_to_generate, cls)
                synthetic_labels = np.full((n_to_generate,), cls)

                X_balanced.append(synthetic_samples)
                y_balanced.append(synthetic_labels)

        X_balanced = np.vstack(X_balanced)
        y_balanced = np.hstack(y_balanced)

        self.logger.info(f"Standardizing balanced data...")
        X_balanced = self.scaler.transform(X_balanced)

        return shuffle(X_balanced, y_balanced, random_state=42)

    def _load_data(self, path, test_size=0.2, standardize=False, stratify=False):
        print('Loading data...')
        data = np.load(path)
        self.X, self.y = data['windows'], data['labels']

        mask = ~np.isnan(self.X).any(axis=1) & ~np.isnan(self.y)
        self.X, self.y = self.X[mask], self.y[mask]

        if standardize:
            print('Standardizing data...')
            self.X = self._standardize_data()

        print('Splitting data...')
        if stratify:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42
            )

    def _standardize_data(self):
        self.scaler = QuantileTransformer(output_distribution='normal')
        standardized = self.scaler.fit_transform(self.X)

        return standardized

    
    def _random_undersample(self):
        class_counts = Counter(self.y_train)
        min_class_size = min(class_counts.values())
        
        balanced_X = []
        balanced_y = []
        
        for cls in class_counts.keys():
            class_indices = np.where(self.y_train == cls)[0]
            sampled_indices = np.random.choice(class_indices, size=min_class_size, replace=False)
            balanced_X.append(self.X_train[sampled_indices])
            balanced_y.append(self.y_train[sampled_indices])
        
        balanced_X = np.vstack(balanced_X)
        balanced_y = np.concatenate(balanced_y)
        
        return balanced_X, balanced_y