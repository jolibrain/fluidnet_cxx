# config.py

# Keep this table in alphabetical order.
conf = {
    'batch_size' : 100,
    'dataDir' : '../data/datasets',
    'dataset' : 'output_current_model_sphere',
    'freq_to_file' : 5,
    'loadModel' : False,  # set to True when resuming training or evaluating
    'maxEpochs' : 2000,
    'modelDir' : 'models',
    'modelFilename' : 'conv_model',  # Output model file name
    'newModel' : {
        'normalizeInputThreshold' :  0.00001,  # Don't normalize input noise.
    },
    'num_workers' : 10,
    'resume_training': True, # Set to True when resuming
    'save_model_dir' : 'model',
    'save_plots_dir' : 'plot',
    'train' : True
}

