from torchvision import transforms
import numpy as np

BASE_DIR = '/mnt/diabetic_retinopathy_v2'

DATA_DIR = BASE_DIR + '/data'

MODEL_DIR = BASE_DIR + '/models'

TRAIN_DIR = DATA_DIR + '/both/train_images_numpy_299'
TEST_DIR = DATA_DIR + '/both/train_images_numpy_299' #Same path because we split train in train and test.

TRAIN_LABELS = DATA_DIR + '/both/train.csv'
TEST_LABELS = DATA_DIR + '/both/test.csv'

BATCH_SIZE = 64

NUM_EPOCH = 200

TEST_SPLIT = 0.3

SEED = 42

NUM_CLASSES = 1

ID_COLUMN = 'id_code'

LABEL_COLUMN = ['diagnosis']

BLACK_LIST_ID = []

INPUT_SIZE = 299

CHANNELS = 3

IMAGE_FORMAT = 'npy'

MODELS = ['ResNet50']

LOAD_CHECKPOINT = False

OPTIMIZERS = ['AmsGradAdam0005']

SCHEDULERS = ['ReduceLROnPlateau'] # Add None if you don't want a scheduler, just optimizer.

LOSSES = ['SmoothL1Loss']

SAMPLE_FRAC = 0.5 #Fraction of dataset to use. Set to 1.0 to use the entire dataset.

CUDA_DEVICES = [0,1,2,3]

METRIC = 'KAPPA' # ACC, KAPPA

SAVE_BEST = 'loss' # [metric | loss]

#TRAIN_AUGMENTATION = [transforms.RandomRotation((0,360)),
#                     transforms.RandomHorizontalFlip(p=0.5),
#                     transforms.RandomVerticalFlip(p=0.5),
#                     transforms.ToTensor(),
#                     transforms.Normalize(np.array([0.485, 0.456, 0.406], dtype=np.float32), 
#                                          np.array([0.229, 0.224, 0.225], dtype=np.float32))]
#
#TEST_AUGMENTATION = [transforms.ToTensor(),
#                    transforms.Normalize(np.array([0.485, 0.456, 0.406], dtype=np.float32), 
#                                         np.array([0.229, 0.224, 0.225], dtype=np.float32))]

# Data Augumentation with Dataset Mean and Standard Deviation
TRAIN_AUGMENTATION = [transforms.RandomRotation((0,360)),
                     transforms.RandomHorizontalFlip(p=0.5),
                     transforms.RandomVerticalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Normalize(np.array([0.432, 0.301, 0.215], dtype=np.float32), 
                                          np.array([0.274, 0.201, 0.167], dtype=np.float32))]

TEST_AUGMENTATION = [transforms.ToTensor(),
                    transforms.Normalize(np.array([0.432, 0.301, 0.215], dtype=np.float32), 
                                         np.array([0.274, 0.201, 0.167], dtype=np.float32))]