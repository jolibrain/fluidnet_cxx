# config.py

defaultConf = {
    'batchSize' : 64,
    'dataDir' : '../fdata',
    'dataset' : 'output_current_model_sphere',
    'freqToFile' : 5,
   # 'loadModel' : False,  # set to True when resuming training or evaluating
    'maxEpochs' : 2000,
    'modelDir' : 'data/model_pLoss_L1_L2',
    'modelFilename' : 'convModel',  # Output model file name
    'modelParam' : {
        'inputChannels' : {
            'div' : False,
            'pDiv': False,
            'UDiv': True,
        },
        'pL2Lambda' : 1,
        'divL2Lambda' : 0,
        'pL1Lambda' : 0.25,
        'divL1Lambda' : 0,
        'normalizeInput' : True,
        'normalizeInputChan' : 'UDiv',
        'normalizeInputThreshold':0.00001,  # Don't normalize input noise.
    },
    'numWorkers' : 3,
    'preprocOnly': False, # Only preprocesses the dataset and exits
    'resumeTraining': True, # Set to True when resuming
    'train' : True
}

