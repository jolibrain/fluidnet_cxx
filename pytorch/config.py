# config.py

defaultConf = {
    'batchSize' : 32,
    'dataDir' : '../fdata', # Dataset dir
    'dataset' : 'output_current_model_sphere', # Folder inside dataDir with tr and te scenes
    'freqToFile' : 5, # Frequency for loss output to file
   # 'loadModel' : False,  # set to True when resuming training or evaluating
    'maxEpochs' : 400,
    'modelDir' : 'data2/model_divL2_5divLT_ScaleNet_New', # Folder where the model, the losses and
                                           # the conf and mconf files are saved
    'modelFilename' : 'convModel',  # Output model file name
    'modelParam' : {
        'inputChannels' : {
            'div' : True,
            'pDiv': False,
            'UDiv': False,
        },
        # Set to 0 to de-activate corresponding loss.
        'pL2Lambda' : 0,
        'divL2Lambda' : 1,
        'pL1Lambda' : 0,
        'divL1Lambda' : 0,
        'divLongTermLambda' : 5,
        'dt' : 0.1,
        'buoyancyScale' : 0,
        'gravityScale' : 0,
        'correctScalar' : False,
        'viscosity': 0,
        'model' : 'ScaleNet', # 'FluidNet' for our own net, based on FluidNet article.
                              # 'ScaleNet' for Multi Scale Net
                              # (similar but deeper than FluidNet)
        'normalizeInput' : True,
        'normalizeInputChan' : 'UDiv',
        'normalizeInputThreshold': 0.00001,  # Don't normalize input noise.
        'longTermDivNumSteps' : [4, 16],
        'longTermDivProbability': 0.9,
        'timeScaleSigma': 1, # Amplitude of time scale perturbation during
                             # during training.
        'maccormackStrength' : 0.6,
        'sampleOutsideFluid' : False,
        'lr' : 5e-5,
    },
    'numWorkers' : 3,
    'preprocOnly': False, # If True, only preprocesses the dataset and exits
    # Debug options
    'printTraining' : 'save', # Save fields while training
                              # 'show' to show plots in interactive session
                              # 'save' to save plots to disk
                              # other string to do neither of it
    'shuffleTraining': True, # For debug printing tr set during training
                              # If True, prints the validation set instead.
    # Resume
    'resumeTraining': False, # Set to True when resuming
}

