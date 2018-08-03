# config.py

defaultConf = {
    'batchSize' : 64,
    'dataDir' : '../fdata',
    'dataset' : 'output_current_model_sphere',
    'freqToFile' : 5, # Frequency for loss output to file
   # 'loadModel' : False,  # set to True when resuming training or evaluating
    'maxEpochs' : 2,
    'modelDir' : 'data/model_pLoss_L1_L2', # Folder where the model, the losses and
                                           # the conf and mconf files are saved
    'modelFilename' : 'convModel',  # Output model file name
    'modelParam' : {
        'inputChannels' : {
            'div' : True,
            'pDiv': False,
            'UDiv': False,
        },
        # Set to 0 to de-activate
        'pL2Lambda' : 0,
        'divL2Lambda' : 1,
        'pL1Lambda' : 0,
        'divL1Lambda' : 0,
        'normalizeInput' : True,
        'normalizeInputChan' : 'UDiv',
        'normalizeInputThreshold':0.00001,  # Don't normalize input noise.
    },
    'numWorkers' : 3,
    'preprocOnly': False, # If True, only preprocesses the dataset and exits
    'resumeTraining': False, # Set to True when resuming
    'train' : True # No usage for now
}

