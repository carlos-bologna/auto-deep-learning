from core.parameters import NUM_CLASSES

LOSS_LIST = {
    'NLLLoss': {
        'function': 'NLLLoss',
        'last_layer': 'logsoftmax',
        'onehotlabel': False,
        'weight': None,
        'size_average': None,
        'ignore_index': -100,
        'reduce': None,
        'reduction': 'mean'
    },
    'MSELoss': {
        'function': 'MSELoss',
        'last_layer': 'linear',
        'onehotlabel': False,
        'weight': None, # will be ignored
        'reduction': 'mean'
    },
    'SmoothL1Loss': {
        'function': 'SmoothL1Loss',
        'onehotlabel': False,
        'last_layer': 'linear',
        'reduction': 'mean',
        'weight': None, # will be ignored
    },
    'CrossEntropyLoss': {
        'function': 'CrossEntropyLoss',
        'onehotlabel': True,
        'last_layer': '?',
        'weight': None,
        'size_average': None,
        'ignore_index': -100,
        'reduce': None,
        'reduction': 'mean'
    },
    'WeightedMultiLabelLogLoss': {
        'function': 'WeightedMultiLabelLogLoss',
        'n_classes': NUM_CLASSES,
        'weight': None
    },
    'WeightedMultiLabelFocalLogLoss': {
        'function': 'WeightedMultiLabelFocalLogLoss',
        'n_classes': NUM_CLASSES,
        'weight': None,
        'gamma': 2
    }
}