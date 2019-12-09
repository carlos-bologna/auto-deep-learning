from torchvision import transforms
import numpy as np

# Source Directories
BASE_DIR = '/mnt/diabetic_retinopathy'
DATA_DIR = BASE_DIR + '/data'
MODEL_DIR = BASE_DIR + '/models'
DST_DATA_DIR = BASE_DIR + '/npy'

CLEAR_ALL_DATA_BEFORE_PREPROCESS = False

# Hiperparameters
CUDA_DEVICES = [-1] #[0,1,2,3] or [2,3] or any combination. Use -1 for all available GPUs.

SEED = 42

NUM_EPOCH = 10

TEST_SPLIT = 0.2

NUM_CLASSES = 1

CHANNELS = 3

INPUT_SIZES = [512]

BATCH_SIZES = [64]

MODELS = ['ResNet101AttentionPre', 'ResNet50AttentionPre']

LOAD_CHECKPOINT = True

OPTIMIZERS = ['Adam0001']

SCHEDULERS = [None] # Add None if you don't want a scheduler, just optimizer.

LOSSES = ['SmoothL1Loss']

SAMPLE_FRACS = [0.5] #Fraction of dataset to use. Set to 1.0 to use the entire dataset.

METRIC = 'KAPPA' # ACC, KAPPA

SAVE_BEST = 'loss' # [metric | loss]

# Data Augumentation with Dataset Mean and Standard Deviation
AUGMENTATION_TAG = 'DatasetMeanStd'

#TRAIN_AUGMENTATION = [transforms.RandomRotation((0,360)),
#                     transforms.RandomHorizontalFlip(p=0.5),
#                     transforms.RandomVerticalFlip(p=0.5),
#                     transforms.ToTensor(),
#                     transforms.Normalize(np.array([0.432, 0.301, 0.215], dtype=np.float32), 
#                                          np.array([0.274, 0.201, 0.167], dtype=np.float32))]
#
#TEST_AUGMENTATION = [transforms.ToTensor(),
#                    transforms.Normalize(np.array([0.432, 0.301, 0.215], dtype=np.float32), 
#                                         np.array([0.274, 0.201, 0.167], dtype=np.float32))]

TRAIN_AUGMENTATION = [transforms.ColorJitter(brightness=0.2, 
                                             contrast=0.2, 
                                             saturation=0.2, 
                                             hue=0.1),
                      transforms.RandomAffine(degrees=0, 
                                             translate=(0.0, 0.2), 
                                             scale=(1.0, 1.2), 
                                             shear=0.2, 
                                             resample=False, 
                                             fillcolor=0),
                     transforms.RandomRotation((0,180)),
                     transforms.RandomHorizontalFlip(p=0.5),
                     transforms.RandomVerticalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Normalize(np.array([0.432, 0.301, 0.215], dtype=np.float32), 
                                          np.array([0.274, 0.201, 0.167], dtype=np.float32))]

TEST_AUGMENTATION = [transforms.ToTensor(),
                    transforms.Normalize(np.array([0.432, 0.301, 0.215], dtype=np.float32), 
                                         np.array([0.274, 0.201, 0.167], dtype=np.float32))]
