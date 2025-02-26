import hashlib
from constants import *
from modules import IIIS_DATA as IIIS

def generate_experiment_name(generative_model, classifier_model, generative_model_args, classifier_args):
    gen_params = '_'.join(f"{key}={value}" for key, value in generative_model_args.items())
    clf_params = '_'.join(f"{key}={value}" for key, value in classifier_args.items())
    full_params = f"{gen_params}_{clf_params}"
    hash_id = hashlib.md5(full_params.encode()).hexdigest()[:8]
    return f"{generative_model}_{classifier_model}_{hash_id}"

def run_experiment(generative_model, classifier_model):
    batch_size = 128
    num_epochs = 100
    lr = 1e-3
    device = 'cuda'

    if generative_model == 'deltaEncoder':
        generative_model_args = {
            'num_epoch': 100,
            'learning_rate': 1e-3,
            'drop_out_rate': 0.3,
            'drop_out_rate_input': 0.2,
            'batch_size': batch_size,
            'noise_size': 64,
            'encoder_size': [128, 64],
            'decoder_size': [64, 128]
        }
    elif generative_model == 'cGan':
        generative_model_args = {
            'device': device,
            'z_size': 1,
            'num_epoch': 100,
            'class_num': 4,
            'batch_size': batch_size,
            'input_size': 512,
            'generator_layer_size': [128, 256, 512],
            'discriminator_layer_size': [512, 256, 128],
            'learning_rate': lr
        }
    elif generative_model == 'wgan':
        generative_model_args = {
            "device": device,
            "num_classes": 4,
            "channels": 1,
            "generator_iters": 100,
            "critic_iter": 5,
            "batch_size": batch_size,
            "latent_dim": 100,
            "learning_rate": 0.00005,
            "weight_cliping_limit": 0.01,
            "num_epoch": 1,
            "generator_conv": [256, 128, 64],
            "discriminator_conv": [64, 128, 256],
            "latent_feature_dim": 256,
            "scheduler_step": 10,
            "scheduler_gamma": 0.95,
            "early_stop_patience": 10,
        }

    if classifier_model == 'simpleClassifier':
        classifier_args = {
            'input_dim': 512,
            'output_dim': 4,
            'epochs': num_epochs,
            'lr': lr,
            'device': device,
            'hidden_dims': [256, 128],
            'batch_size': batch_size
        }
    elif classifier_model == 'transformer':
        nhead = 2
        d_model = 32
        if d_model % nhead != 0:
            d_model += nhead - (d_model % nhead)
            print(f"Adjusted d_model to {d_model} for compatibility with nhead={nhead}")
        classifier_args = {
            'batch_size': batch_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': 2,
            'dim_feedforward': 128,
            'dropout': 0.2,
            'lr': lr,
            'num_epochs': num_epochs,
            'device': device
        }
    elif classifier_model == 'eegnet':
        classifier_args = { 
            'F1': 16,
            'D': 4,
            'Samples': 512,
            'kernLength': 10,
            'nb_classes': 4,
            'dropoutRate': 0.5,
            'batch_size': batch_size,
            'epochs': num_epochs,
            'learning_rate': lr,
            'device': device
        }
    elif classifier_model == 'squeezeformer':
        classifier_args = {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr,
            'lr_decay_step': 4,
            'lr_decay_gamma': 0.1,
            'model_size': 'small',
            'device': device
        }

    config = BASE_CONFIG.copy()
    config.update({
        'generative_model': generative_model,
        'generative_model_args': generative_model_args,
        'classifier_model': classifier_model,
        'classifier_args': classifier_args,
        'experiment_name': generate_experiment_name(generative_model, classifier_model, generative_model_args, classifier_args)
    })

    print(f"Running experiment: {config['experiment_name']}")
    iiis_data_pipeline = IIIS(config)
    
    metrics_before, metrics_after = iiis_data_pipeline.process()
    print(f"Results for {generative_model} with {classifier_model}:")
    print(f"  Metrics before balancing: {metrics_before}")
    print(f"  Metrics after balancing:  {metrics_after}\n")

if __name__ == "__main__":
    generative_models = ['deltaEncoder', 'cGan', 'wgan']
    classifier_models = ['squeezeformer']


    for gm in generative_models:
        for cm in classifier_models:
            run_experiment(gm, cm)
