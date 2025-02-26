import optuna
import hashlib
from constants import *
from modules.IIIS_Data import IIIS_DATA as IIIS

def generate_experiment_name(generative_model, classifier_model, generative_model_args, classifier_args):
    gen_params = '_'.join(f"{key}={value}" for key, value in generative_model_args.items())
    clf_params = '_'.join(f"{key}={value}" for key, value in classifier_args.items())
    full_params = f"{gen_params}_{clf_params}"
    hash_id = hashlib.md5(full_params.encode()).hexdigest()[:8]
    return f"{generative_model}_{classifier_model}_{hash_id}"


def objective(trial, generative_model, classifier_model):
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    if generative_model == 'deltaEncoder':
        generative_model_args = {
            # 'num_epoch': trial.suggest_int('num_epoch', 50, 150),
            'num_epoch': trial.suggest_int('num_epoch', 50, 100),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'drop_out_rate': trial.suggest_uniform('drop_out_rate', 0.2, 0.5),
            'drop_out_rate_input': trial.suggest_uniform('drop_out_rate_input', 0.1, 0.3),
            'batch_size': batch_size,
            'noise_size': trial.suggest_int('noise_size', 32, 128),
            'encoder_size': trial.suggest_categorical('encoder_size', [[128, 64], [256, 128]]),
            'decoder_size': trial.suggest_categorical('decoder_size', [[64, 128], [128, 256]])
        }
    elif generative_model == 'cGan':
        generative_model_args = {
            'device': 'cuda',
            'z_size': trial.suggest_int('z_size', 1, 1),
            # 'num_epoch': trial.suggest_int('num_epoch', 20, 100),
            'num_epoch': trial.suggest_int('num_epoch', 20, 100),
            'class_num': 4,
            'batch_size': batch_size,
            'input_size': 512,
            'generator_layer_size': trial.suggest_categorical('generator_layer_size', [[64, 128, 256], [128, 256, 512]]),
            'discriminator_layer_size': trial.suggest_categorical('discriminator_layer_size', [[256, 128, 64], [512, 256, 128]]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        }
    else:
        raise ValueError("Invalid generative model")

    if classifier_model == 'simpleClassifier':
        classifier_args = {
            'input_dim': 512,
            'output_dim': 4,
            # 'epochs': trial.suggest_int('epochs', 10, 50),
            'epochs': trial.suggest_int('epochs', 10, 50),
            'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
            'device': 'cuda',
            'hidden_dims': trial.suggest_categorical('hidden_dims', [[128, 64], [256, 128]]),
            'batch_size': batch_size
        }

    elif classifier_model == 'transformer':
        nhead = trial.suggest_int('nhead', 1, 4)
        d_model = trial.suggest_int('d_model', 32, 64)

        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model += nhead - (d_model % nhead)  # Adjust d_model to the nearest divisible value
            print(f"Adjusted d_model to {d_model} for compatibility with nhead={nhead}")

        classifier_args = {
            'batch_size': batch_size,
            'd_model': d_model,  # Updated
            'nhead': nhead,
            'num_encoder_layers': trial.suggest_int('num_encoder_layers', 1, 4),
            'dim_feedforward': trial.suggest_int('dim_feedforward', 64, 256),
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
            'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
            'num_epochs': trial.suggest_int('num_epochs', 20, 50),
            'device': 'cuda'
        }

    else:
        raise ValueError("Invalid classifier model")

    config = BASE_CONFIG.copy()
    config.update({
        'generative_model': generative_model,
        'generative_model_args': generative_model_args,
        'classifier_model': classifier_model,
        'classifier_args': classifier_args,
        'experiment_name': generate_experiment_name(generative_model, classifier_model, generative_model_args, classifier_args)
    })

    iiis_data_pipeline = IIIS(config)

    try:
        metrics_before, metrics_after = iiis_data_pipeline.process()
        return metrics_after['f1']
    except Exception as e:
        print(f"Error running configuration for {generative_model} with {classifier_model}: {str(e)}")
        return 0.0

for generative_model in ['cGan']:
    for classifier_model in ['transformer']:
        study_name = f"{generative_model}_{classifier_model}_optimization"
        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(lambda trial: objective(trial, generative_model, classifier_model), n_trials=5)

        print(f"Best hyperparameters for {generative_model} with {classifier_model}:")
        best_params = study.best_params
        print(best_params)

        best_config = BASE_CONFIG.copy()
        best_config.update({
            'generative_model': generative_model,
            'classifier_model': classifier_model,
            'generative_model_args': {k: v for k, v in best_params.items() if k in CGAN_ARGS or DELTA_ARGS},
            'classifier_args': {k: v for k, v in best_params.items() if k in SIMPLE_CLASSIFIER_ARGS or TRANSFORMER_ARGS},
            'experiment_name': f"{generative_model}_{classifier_model}_bestParams"
        })

        iiis_best_pipeline = IIIS(best_config)
        try:
            metrics_before, metrics_after = iiis_best_pipeline.process()
            print(f"Metrics before balancing for {generative_model} with {classifier_model}: {metrics_before}")
            print(f"Metrics after balancing for {generative_model} with {classifier_model}: {metrics_after}")
        except Exception as e:
            print(f"Error running best configuration for {generative_model} with {classifier_model}: {str(e)}")