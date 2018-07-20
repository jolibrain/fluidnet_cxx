# config.py

defaultConf = {
    'batchSize' : 100,
    'dataDir' : '../data/datasets',
    'dataset' : 'output_current_model_sphere',
    'freqToFile' : 5,
   # 'loadModel' : False,  # set to True when resuming training or evaluating
    'maxEpochs' : 2000,
    'modelDir' : 'data/model_div_deconv_cat',
    'modelFilename' : 'convModel',  # Output model file name
    'modelParam' : {
        'inputChannels' : {
            'div' : True,
            'pDiv': False,
            'UDiv': False,
        },
        'lossP' : False,
        'lossU' : False,
        'lossDiv' : True,
        'normalizeInput' : True,
        'normalizeInputChan' : 'UDiv',
        'normalizeInputThreshold':0.00001,  # Don't normalize input noise.
    },
    'numWorkers' : 12,
    'preprocOnly': False, # Only preprocesses the dataset and exits
    'resumeTraining': False, # Set to True when resuming
    'train' : True
}

