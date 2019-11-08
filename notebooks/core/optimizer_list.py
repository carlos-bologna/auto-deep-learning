OPTIMIZER_LIST = {
    'DefaultAdam': {
        'function': 'Adam',
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'eps': 1e-08,
        'weight_decay': 0,
        'amsgrad': False
    },
    'HalfLRAdam': {
        'function': 'Adam',
        'lr': 0.0005,
        'betas': (0.9, 0.999),
        'eps': 1e-08,
        'weight_decay': 0,
        'amsgrad': False
    },
    'DoubleWDAdam': {
        'function': 'Adam',
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'eps': 1e-08,
        'weight_decay': 0.0001,
        'amsgrad': False
    },
    'SGDMomentum': {
        'function': 'SGD',
        'lr': 0.1,
        'weight_decay': 0,
        'momentum': 0.01
    },
    'SGDMomentumV2': {
        'function': 'SGD',
        'lr': 0.05,
        'weight_decay': 0,
        'momentum': 0.01
    },
    'SGDMomentumV3': {
        'function': 'SGD',
        'lr': 0.1,
        'weight_decay': 0,
        'momentum': 0.005
    },
    'SGDMomentumV4': {
        'function': 'SGD',
        'lr': 0.05,
        'weight_decay': 0,
        'momentum': 0.005
    },
    'SGDMomentumV5': {
        'function': 'SGD',
        'lr': 0.03,
        'weight_decay': 0,
        'momentum': 0.01
    },
    'SGDMomentumV6': {
        'function': 'SGD',
        'lr': 0.01,
        'weight_decay': 0,
        'momentum': 0.01
    },
    'SGDMomentumV7': {
        'function': 'SGD',
        'lr': 0.0075,
        'weight_decay': 0,
        'momentum': 0.01
    }
}