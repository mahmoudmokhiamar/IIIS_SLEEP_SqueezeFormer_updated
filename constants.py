BASE_CONFIG = {
    'data_path': '/home/mahmoudmokhiamar/Documents/IIISSleep/A_128Hz4sec4stage_output_combined_EEG_data_nonan_windowslabels.npz',
    'balance': True,
    'test_size': 0.2,
    'standardize': True,
    'k_folds': 2,
}

CGAN_ARGS = {
    'device': 'cuda',
    'z_size': 64,
    'num_epoch': 50,
    'class_num': 4,
    'batch_size': 256,
    'input_size': 512,
    'generator_layer_size': [128, 256, 512],
    'discriminator_layer_size': [512, 256, 128],
    'learning_rate': 1e-3
}

DELTA_ARGS = {
    'num_epoch': 100,
    'learning_rate': 1e-3, 
    'drop_out_rate': 0.4,
    'drop_out_rate_input': 0.2,
    'batch_size': 32,
    'noise_size' : 64,
    'encoder_size' : [256, 128],
    'decoder_size' : [128, 256]
}

SIMPLE_CLASSIFIER_ARGS = {
    'input_dim': 512,
    'output_dim': 4,
    'epochs': 20,
    'lr': 0.001,
    'device': 'cuda',
    'hidden_dims': [256, 128],
    'batch_size': 64
}

TRANSFORMER_ARGS = {
    'batch_size': 256,
    'd_model': 64,
    'nhead': 4,
    'num_encoder_layers': 2,
    'dim_feedforward': 128,
    'dropout': 0.1,
    'lr': 1e-3,
    'num_epochs': 20,
    'device': 'cuda'
}