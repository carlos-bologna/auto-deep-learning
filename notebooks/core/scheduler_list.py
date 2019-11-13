SCHEDULER_LIST = {
    'ReduceLROnPlateau': {
        'function': 'ReduceLROnPlateau',
        'mode': 'min', 
        'factor': 0.1, 
        'patience': 10, 
        'verbose': False, 
        'threshold': 0.0001, 
        'threshold_mode': 'rel', 
        'cooldown': 0, 
        'min_lr': 0, 
        'eps': 1e-08
    }
}