from torchvision import transforms

BASE_DIR = '/mnt/diabetic_retinopathy_v2'

DATA_DIR = BASE_DIR + '/data'

MODEL_DIR = BASE_DIR + '/models'

TRAIN_DIR = DATA_DIR + '/both/train_images_numpy_299'
TEST_DIR = DATA_DIR + '/both/train_images_numpy_299' #Same path because we split train in train and test.

TRAIN_LABELS = DATA_DIR + '/both/train.csv'
TEST_LABELS = DATA_DIR + '/both/test.csv'

BATCH_SIZE = 64

NUM_EPOCH = 50

TEST_SPLIT = 0.3

SEED = 42

NUM_CLASSES = 1

ID_COLUMN = 'id_code'

LABEL_COLUMN = ['diagnosis']

BLACK_LIST_ID = []

INPUT_SIZE = 299

IMAGE_FORMAT = 'npy'

MODELS = ['ResNet50']

LOAD_CHECKPOINT = True

CHECKPOINTS = {'resnet50': 'FineTuningResNet50Continue_DefaultAdam_MSELoss_imgsize299_loss_0.3881308227288331.pt'}

OPTIMIZERS = ['DefaultAdam']

LOSSES = ['SmoothL1Loss']

SAMPLE_FRAC = 1.0 #Fraction of dataset to use. Set to 1.0 to use the entire dataset.

CUDA_DEVICES = [0]

METRIC = 'KAPPA' # ACC, KAPPA

SAVE_BEST = 'loss' # [metric | loss]

TRAIN_AUGMENTATION = [transforms.RandomRotation((0,360)),
                      transforms.RandomHorizontalFlip(p=0.5),
                      transforms.RandomVerticalFlip(p=0.5),
                      transforms.ToTensor()]

TEST_AUGMENTATION = [transforms.ToTensor()]