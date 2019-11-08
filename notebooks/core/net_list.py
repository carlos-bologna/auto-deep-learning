NET_LIST = {
    'FineTuningDensenet121MultiTask': {
        'base_model': 'densenet121multitask',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': []
    },
    'FineTuningDensenet121MultiTaskV2': {
        'base_model': 'densenet121multitaskV2',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': []
    },
    'Densenet121': {
        'base_model': 'densenet121',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': []
    },
    'FineTuningDensenet121v1': {
        'base_model': 'densenet121',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': ['conv0', 'norm0', 'denseblock1', 'transition1', 
                             'denseblock2', 'transition2', 'denseblock3', 'transition3', 
                             'denseblock4', 'norm5']
    },
    'FineTuningDensenet121v2': {
        'base_model': 'densenet121',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': ['conv0', 'norm0', 'denseblock1', 'transition1', 
                             'denseblock2', 'transition2', 'denseblock3', 'transition3']
    },
    'FineTuningDensenet121v3': {
        'base_model': 'densenet121',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': ['conv0', 'norm0', 'denseblock1', 'transition1', 
                             'denseblock2', 'transition2']
    },
    'VGG16': {
        'base_model': 'vgg16',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': []
    },
    'FineTuningVGG16v1': {
        'base_model': 'vgg16',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': ['features.0', 'features.2', 'features.5', 'features.7', 
                             'features.10', 'features.12', 'features.14', 'features.17', 
                             'features.19', 'features.21', 'features.24', 'features.26', 
                             'features.28', 'classifier.0', 'classifier.3']
    },
    'ResNet50': {
        'base_model': 'resnet50',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': []
    },
    'FineTuningResNet50v1': {
        'base_model': 'resnet50',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
    },
    'FineTuningResNet50v2': {
        'base_model': 'resnet50',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
    },
    'FineTuningResNet50v3': {
        'base_model': 'resnet50',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': ['conv1', 'bn1', 'layer1', 'layer2']
    },    
    'ResNet50Attention': {
        'base_model': 'ResNet50Attention',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': []
    },
    'FineTuningResNet50AttentionMultiTask': {
        'base_model': 'ResNet50AttentionMultiTask',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': []
    },    
    'FineTuningResNet50AttentionMultiTaskV2': {
        'base_model': 'ResNet50AttentionMultiTaskV2',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': []
    },
    'Inception3': {
        'base_model': 'inception_v3',
        'is_inception': True,
        'pretrained': True,
        'layers_to_frozen': []
    },
    'FineTuningInception3v1': {
        'base_model': 'inception_v3',
        'is_inception': True,
        'pretrained': True,
        'layers_to_frozen': ['AuxLogits.conv0', 'AuxLogits.conv1', 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 
                             'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 
                             'Mixed_5b.branch1x1', 'Mixed_5b.branch3x3dbl_1', 
                             'Mixed_5b.branch3x3dbl_2', 'Mixed_5b.branch3x3dbl_3', 'Mixed_5b.branch5x5_1', 
                             'Mixed_5b.branch5x5_2', 'Mixed_5b.branch_pool', 'Mixed_5c.branch1x1', 
                             'Mixed_5c.branch3x3dbl_1', 'Mixed_5c.branch3x3dbl_2', 'Mixed_5c.branch3x3dbl_3', 
                             'Mixed_5c.branch5x5_1', 'Mixed_5c.branch5x5_2', 'Mixed_5c.branch_pool', 
                             'Mixed_5d.branch1x1', 'Mixed_5d.branch3x3dbl_1', 'Mixed_5d.branch3x3dbl_2', 
                             'Mixed_5d.branch3x3dbl_3', 'Mixed_5d.branch5x5_1', 'Mixed_5d.branch5x5_2', 
                             'Mixed_5d.branch_pool', 'Mixed_6a.branch3x3', 'Mixed_6a.branch3x3dbl_1', 
                             'Mixed_6a.branch3x3dbl_2', 'Mixed_6a.branch3x3dbl_3', 'Mixed_6b.branch1x1', 
                             'Mixed_6b.branch7x7_1', 'Mixed_6b.branch7x7_2', 'Mixed_6b.branch7x7_3', 
                             'Mixed_6b.branch7x7dbl_1', 'Mixed_6b.branch7x7dbl_2', 'Mixed_6b.branch7x7dbl_3', 
                             'Mixed_6b.branch7x7dbl_4', 'Mixed_6b.branch7x7dbl_5', 'Mixed_6b.branch_pool', 
                             'Mixed_6c.branch1x1', 'Mixed_6c.branch7x7_1', 'Mixed_6c.branch7x7_2', 
                             'Mixed_6c.branch7x7_3', 'Mixed_6c.branch7x7dbl_1', 'Mixed_6c.branch7x7dbl_2', 
                             'Mixed_6c.branch7x7dbl_3', 'Mixed_6c.branch7x7dbl_4', 'Mixed_6c.branch7x7dbl_5', 
                             'Mixed_6c.branch_pool', 'Mixed_6d.branch1x1', 'Mixed_6d.branch7x7_1', 
                             'Mixed_6d.branch7x7_2', 'Mixed_6d.branch7x7_3', 'Mixed_6d.branch7x7dbl_1', 
                             'Mixed_6d.branch7x7dbl_2', 'Mixed_6d.branch7x7dbl_3', 'Mixed_6d.branch7x7dbl_4', 
                             'Mixed_6d.branch7x7dbl_5', 'Mixed_6d.branch_pool', 'Mixed_6e.branch1x1', 
                             'Mixed_6e.branch7x7_1', 'Mixed_6e.branch7x7_2', 'Mixed_6e.branch7x7_3', 
                             'Mixed_6e.branch7x7dbl_1', 'Mixed_6e.branch7x7dbl_2', 'Mixed_6e.branch7x7dbl_3', 
                             'Mixed_6e.branch7x7dbl_4', 'Mixed_6e.branch7x7dbl_5', 'Mixed_6e.branch_pool', 
                             'Mixed_7a.branch3x3_1', 'Mixed_7a.branch3x3_2', 'Mixed_7a.branch7x7x3_1', 
                             'Mixed_7a.branch7x7x3_2', 'Mixed_7a.branch7x7x3_3', 'Mixed_7a.branch7x7x3_4', 
                             'Mixed_7b.branch1x1', 'Mixed_7b.branch3x3_1', 'Mixed_7b.branch3x3_2a', 
                             'Mixed_7b.branch3x3_2b', 'Mixed_7b.branch3x3dbl_1', 'Mixed_7b.branch3x3dbl_2', 
                             'Mixed_7b.branch3x3dbl_3a', 'Mixed_7b.branch3x3dbl_3b', 'Mixed_7b.branch_pool', 
                             'Mixed_7c.branch1x1', 'Mixed_7c.branch3x3_1', 'Mixed_7c.branch3x3_2a', 
                             'Mixed_7c.branch3x3_2b', 'Mixed_7c.branch3x3dbl_1', 'Mixed_7c.branch3x3dbl_2', 
                             'Mixed_7c.branch3x3dbl_3a', 'Mixed_7c.branch3x3dbl_3b', 'Mixed_7c.branch_pool']
    },
    'EfficientNetB7': {
        'base_model': 'efficientnetb7',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': []
    },
    'FineTuningEfficientNetB7v1': {
        'base_model': 'efficientnetb7',
        'is_inception': False,
        'pretrained': True,
        'layers_to_frozen': ['_conv_stem', '_bn0', '_blocks', '_conv_head', '_bn1']
    }
}