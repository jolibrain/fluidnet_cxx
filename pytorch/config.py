# config.py

defaultConf = {
    'batchSize' : 100,
    'dataDir' : '../data/datasets',
    'dataset' : 'output_current_model_sphere',
    'freqToFile' : 1,
    'loadModel' : False,  # set to True when resuming training or evaluating
    'maxEpochs' : 2000,
    'modelDir' : 'data/model_div',
    'modelFilename' : 'convModel',  # Output model file name
    'modelParam' : {
        'inputChannels' : {
            'div' : False,
            'pDiv': False,
            'UDiv': True,
        },
        'lossP' : True,
        'lossU' : False,
        'lossDiv' : False,
        'normalizeInput' : True,
        'normalizeInputChan' : 'UDiv',
        'normalizeInputThreshold':0.00001,  # Don't normalize input noise.
    },
    'numWorkers' : 12,
    'plotDir' : 'data/plot_div',
    'preprocOnly': False, # Only preprocesses the dataset and exits
    'resumeTraining': False, # Set to True when resuming
    'train' : True
}

