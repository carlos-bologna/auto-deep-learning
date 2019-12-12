#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[1]:

# Internal Packages
from nets.ResNet50Attention import ResNet50Attention
from nets.ResNet101Attention import ResNet101Attention
from common.myfunctions import plot_confusion_matrix
from common.customloss import QuadraticKappa, WeightedMultiLabelLogLoss, WeightedMultiLabelFocalLogLoss
import common.weights_initialization as w_init
import preprocess.preprocess as prep

# Base Packages
import os
import glob
import json
import copy
import time
import pandas as pd
import numpy as np
from PIL import Image
#import pydicom

# Torch Packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Torchvision Packages
import torchvision.transforms.functional as TF
from torchvision import transforms, utils, datasets
from torchvision.models import densenet121, vgg16, resnet50, resnet101, inception_v3

# Miscellaneous Packages
from efficientnet_pytorch import EfficientNet
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.utils import class_weight
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')



# # Summary

# In[3]:


def PrintCombinations(parameters):
    comb = len(parameters['input_sizes']) * len(parameters['sample_fracs']) * len(parameters['batch_sizes']) * len(parameters['models']) * len(parameters['optimizers']) * len(parameters['schedulers']) * len(parameters['losses'])
    print('Total Combinations:', comb)
    print()
    i=1

    for inp in parameters['input_sizes']:
        for frac in parameters['sample_fracs']:
            for bch in parameters['batch_sizes']:
                for m in parameters['models']:
                    for o in parameters['optimizers']:
                        for s in parameters['schedulers']:
                            for l in parameters['losses']:
                                model_name = f'{i}\n Input Size: {str(inp)}\n Dataset Frac.: {str(frac)}\n Batch Size: {str(bch)}\n Model: {m}\n Scheduler: {s}\n Optimizer: {o}\n Loss: {l}\n'
                                print(model_name)
                                i += 1

# Cuda
def getCudaDevices(parameters):

    is_cuda = torch.cuda.is_available()
    gpu_list = []

    if is_cuda: #GPU

        if parameters['cuda_devices'][0] == -1: # All GPUs
            gpu_list = list(range(0, torch.cuda.device_count()))

        cuda_list = ','.join([str(c) for c in gpu_list])

        device = torch.device("cuda:{}".format(cuda_list))

    else: #CPU
        device = "cpu"

    # Set seed for CUDA (all GPU)
    #torch.cuda.manual_seed_all(SEED)
    return is_cuda, gpu_list, device

# Custom Dataset
class CustomDataset(Dataset):

    def __init__(self, data_dir, test_split, sample_frac, input_size, dst_dir,
                 seed, transform=None, phase='train', clear_cache=False):

        self.input_size = input_size
        self.transform = transform
        self.dst_dir = dst_dir
        self.x = []
        self.y = []

        ids = []
        labels = []

        # Load IDs and Labels from directories
        for d in os.listdir(data_dir):

            img_list = os.listdir(os.path.join(data_dir, d))
            ids.extend(img_list)
            labels.extend([d] * len(img_list))

        x_train, x_test, y_train, y_test = train_test_split(ids, labels, test_size = test_split, random_state = seed)

        # Sample Train Dataset
        if sample_frac < 1.0:

            df = pd.DataFrame({'x': x_train, 'y': y_train})

            df_sample = df.sample(frac = sample_frac, random_state=seed)

            x_train = df_sample['x'].tolist()
            y_train = df_sample['y'].tolist()

        # Check Object Phase
        if phase == 'train':
            self.x = x_train
            self.y = y_train
        elif phase == 'test':
            self.x = x_test
            self.y = y_test

        # Check for Preprocess Images
        prep.Preprocess(data_dir, self.x, self.y, input_size, clear_cache, dst_dir)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        img_name = os.path.join(self.dst_dir, str(self.input_size), str(self.y[idx]), self.x[idx].split('.')[0] + '.npy')

        image = np.load(img_name)

        label = int(self.y[idx])

        if self.transform:

           image = self.transform(TF.to_pil_image(image))

        return (image,label)

# Augmentation
def getAugmentation(transform):

    transf_list = []

    for t in transform:

        transf_name = list(t.keys())[0]

        if transf_name == 'ColorJitter':

            transf_list.append(
                transforms.ColorJitter(
                    brightness=t['ColorJitter']['brightness'],
                    contrast=t['ColorJitter']['contrast'],
                    saturation=t['ColorJitter']['saturation'],
                    hue=t['ColorJitter']['hue']))

        if transf_name == 'RandomAffine':

            transf_list.append(
                transforms.RandomAffine(
                    degrees=t['RandomAffine']['degrees'],
                    translate=tuple(t['RandomAffine']['translate']),
                    scale=tuple(t['RandomAffine']['scale']),
                    shear=t['RandomAffine']['shear'],
                    resample=t['RandomAffine']['resample'],
                    fillcolor=t['RandomAffine']['fillcolor']))

        if transf_name == 'RandomRotation':
            transf_list.append(
                transforms.RandomRotation(tuple(t['RandomRotation']))
            )

        if transf_name == 'RandomHorizontalFlip':
            transf_list.append(
                transforms.RandomHorizontalFlip(p=t['RandomHorizontalFlip'])
            )

        if transf_name == 'RandomVerticalFlip':
            transf_list.append(
                transforms.RandomVerticalFlip(p=t['RandomVerticalFlip'])
            )

        if transf_name == 'ToTensor':
            transf_list.append(
                transforms.ToTensor()
            )

        if transf_name == 'Normalize':
            transf_list.append(
                transforms.Normalize(
                    np.array(t['Normalize']['mean'], dtype=np.float32),
                    np.array(t['Normalize']['std'], dtype=np.float32))
            )

    return transf_list


# # Data Loader

# In[7]:


def getDataLoaders(input_size, sample_frac, batch_size, parameters):

    train_transf = transforms.Compose(
        getAugmentation(parameters['data_augmentation']['train'])
    )

    test_transf = transforms.Compose(
        getAugmentation(parameters['data_augmentation']['test'])
    )

    train_dataset = CustomDataset(parameters['directory']['data'],
                                  parameters['test_split'],
                                  sample_frac,
                                  input_size,
                                  parameters['directory']['numpy'],
                                  parameters['seed'],
                                  transform=train_transf,
                                  phase='train',
                                  clear_cache=parameters['clear_all_data_before_preprocess'])


    test_dataset = CustomDataset(parameters['directory']['data'],
                                  parameters['test_split'],
                                  sample_frac,
                                  input_size,
                                  parameters['directory']['numpy'],
                                  parameters['seed'],
                                  transform=test_transf,
                                  phase='test',
                                  clear_cache=parameters['clear_all_data_before_preprocess'])

    # Garregando os dados
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Make a dict to pass though train function
    dataloaders_dict = {'train': train_loader, 'val': test_loader}

    return dataloaders_dict



# # Model

# In[9]:


def getModel(net_list, model_name, general_parameters):

    model_parameters = net_list[model_name]
    base_model = model_parameters['base_model']
    pretrained = model_parameters['pretrained']

    is_cuda, gpu_list, device = getCudaDevices(general_parameters)

    if base_model=='densenet121':

        model = densenet121(pretrained = pretrained)
        model.classifier = nn.Linear(1024, general_parameters['num_classes'])

    elif base_model=='densenet121multitask':

        model = densenet121multitask(pretrained = pretrained)
        model.classifier = nn.Linear(1024, general_parameters['num_classes'])
        model.aux_classifier = nn.Linear(1024, 1)

    elif base_model=='vgg16':

        model = vgg16(pretrained = pretrained)
        model.classifier[6] = nn.Linear(4096, general_parameters['num_classes'])

    elif base_model=='resnet50':

        model = resnet50(pretrained = pretrained)
        model.fc = nn.Linear(2048, general_parameters['num_classes'])

    elif base_model=='resnet101':

        model = resnet101(pretrained = pretrained)
        model.fc = nn.Linear(2048, general_parameters['num_classes'])

    elif base_model=='ResNet50Attention':
        model = ResNet50Attention(general_parameters['num_classes'],
                                  attention=True,
                                  pretrained = pretrained)

    elif base_model=='ResNet101Attention':
        model = ResNet101Attention(general_parameters['num_classes'],
                                  attention=True,
                                  pretrained = pretrained)

    elif base_model=='ResNet50AttentionMultiTask':
        model = ResNet50AttentionMultiTask(general_parameters['num_classes'],
                                  attention=True,
                                  pretrained = pretrained)

    elif base_model=='inception_v3':

        model = inception_v3(pretrained = pretrained)
        model.fc = nn.Linear(2048, general_parameters['num_classes'])
        model.AuxLogits.fc = nn.Linear(768, general_parameters['num_classes'])

    elif base_model=='efficientnetb7':

        model = EfficientNet.from_pretrained('efficientnet-b7')
        model._fc = nn.Linear(2560, general_parameters['num_classes'])

    # Parallel
    # Obs.: when load model, the DataParallel is already in the model.
    if is_cuda & (len(gpu_list) > 1) & (not model_parameters['is_inception']):

        if not general_parameters['cuda_devices']:
            print("Let's use", len(gpu_list), "GPUs!")
            model = nn.DataParallel(model)
        else:
            print("Let's use", general_parameters['cuda_devices'], "GPUs!")
            model = nn.DataParallel(model, device_ids = gpu_list) # When load checkpoint, the DataParallel is already in the model.

    # Frozen Layers
    for name, param in model.named_parameters():
        for l in model_parameters['layers_to_frozen']:
            if l in name:
                param.requires_grad = False

    if general_parameters['load_checkpoint']:

        # Get lastest model file
        list_of_files = glob.glob(general_parameters['directory']['model'] + f'/{base_model}_*.pt') # * means all if need specific format then *.csv

        if len(list_of_files) > 0:

            latest_file = max(list_of_files, key=os.path.getctime)

            print(f'Loading state dict from checkpoint \n\t {latest_file}')

            model.load_state_dict(torch.load(latest_file, map_location=device))
    else:

        if not pretrained:
            model.apply(w_init.weight_init) #Custom weight initialization

    if is_cuda:
        model = model.to(device)

    return model


# # Scheduler

# In[10]:


def getScheduler(scheduler_list, scheduler_name, optimizer):

    if not scheduler_name:
        return None

    scheduler_parameters = scheduler_list[scheduler_name]

    if scheduler_parameters['function'] == 'ReduceLROnPlateau':

        scheduler = ReduceLROnPlateau(optimizer,
                                      mode = scheduler_parameters['mode'],
                                      factor = scheduler_parameters['factor'],
                                      patience = scheduler_parameters['patience'],
                                      verbose = scheduler_parameters['verbose'],
                                      threshold = scheduler_parameters['threshold'],
                                      threshold_mode = scheduler_parameters['threshold_mode'],
                                      cooldown = scheduler_parameters['cooldown'],
                                      min_lr = scheduler_parameters['min_lr'],
                                      eps = scheduler_parameters['eps'])

    return scheduler


# # Optimizer

# In[11]:


def getOptimizer(optimizer_list, optimizer_name, model):

    params_to_update = []

    for name, param in model.named_parameters():

        if param.requires_grad == True:

            params_to_update.append(param)

            #print("\t",name)

    opt_parameters = optimizer_list[optimizer_name]

    if opt_parameters['function'] == 'Adam':

        optimizer = torch.optim.Adam(params_to_update,
                                     lr = opt_parameters['lr'],
                                     betas = tuple(opt_parameters['betas']),
                                     eps = opt_parameters['eps'],
                                     weight_decay = opt_parameters['weight_decay'],
                                     amsgrad = opt_parameters['amsgrad']
                                    )
    elif opt_parameters['function'] == 'SGD':

        optimizer = torch.optim.SGD(params_to_update,
                                     lr = opt_parameters['lr'],
                                     weight_decay = opt_parameters['weight_decay'],
                                     momentum = opt_parameters['momentum']
                                    )

    return optimizer


# # Loss Function

# In[12]:


def getLossFunction(loss_list, loss_nme):

    loss_parameters = loss_list[loss_nme]

    if loss_parameters['function'] == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss(
            reduction = loss_parameters['reduction']
        )

    elif loss_parameters['function'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(
            weight = loss_parameters['weight'],
            size_average = loss_parameters['size_average'],
            ignore_index = loss_parameters['ignore_index'],
            reduce = loss_parameters['reduce'],
            reduction = loss_parameters['reduction']
        )

    elif loss_parameters['function'] == 'NLLLoss':

        criterion = nn.NLLLoss(
            weight = loss_parameters['weight'],
            size_average = loss_parameters['size_average'],
            ignore_index = loss_parameters['ignore_index'],
            reduce = loss_parameters['reduce'],
            reduction = loss_parameters['reduction']
        )

    elif loss_parameters['function'] == 'QuadraticKappa':
        criterion = QuadraticKappa(
            n_classes = loss_parameters['n_classes']
        )

    elif loss_parameters['function'] == 'WeightedMultiLabelLogLoss':

        criterion = WeightedMultiLabelLogLoss(
            n_classes = loss_parameters['n_classes'],
            weight = loss_parameters['weight']
        )
    elif loss_parameters['function'] == 'WeightedMultiLabelFocalLogLoss':

        criterion = WeightedMultiLabelFocalLogLoss(
            n_classes = loss_parameters['n_classes'],
            weight = loss_parameters['weight'],
            gamma = loss_parameters['gamma']
        )

    return criterion

def onehot(labels, num_classes):
    return torch.zeros(len(labels), num_classes).scatter_(1, labels.unsqueeze(1).cpu(), 1.).cuda()


def calcLoss(loss_list, criterion, loss_name, outputs, labels, num_classes):

    loss_parameters = loss_list[loss_name]
    last_layer = loss_parameters['last_layer']

    if last_layer == 'softmax':
        outputs = torch.softmax(outputs, dim=1)
        preds_loss = torch.argmax(outputs, 1)
        preds_metric = torch.argmax(outputs, 1)

    elif last_layer == 'logsoftmax':
        logsoftmax = nn.LogSoftmax(dim=1)
        outputs = logsoftmax(outputs)
        preds_loss = outputs
        preds_metric = torch.argmax(torch.exp(outputs),  1) ### AINDA NÃO TESTADO.

        #OBS.: torch.exp(outputs) revert log

    elif last_layer == 'sigmoid':
        outputs = torch.sigmoid(outputs)
        preds_loss = outputs > 0.5
        preds_metric = torch.argmax(outputs, 1)

    elif last_layer == 'linear':
        preds_loss = outputs
        preds_metric = outputs
        labels = labels.type(torch.float)

    # Transform label from shape 1 to (1, n_classes)
    if loss_parameters['onehotlabel']:
        labels = onehot(labels, num_classes)

    loss = criterion(preds_loss, labels)

    return loss, preds_metric


# # Metric Function

# In[13]:


def calcMetric(preds, labels, metric):

    if metric == 'KAPPA':
        preds = np.round(preds)
        score = cohen_kappa_score(preds, labels, weights='quadratic')

    elif metric == 'ACC':
        score = sum(preds == labels)

    return score


# # Log Data

# In[14]:


def getLogData(log_file, save_best, metric):

    try:

        ea = event_accumulator.EventAccumulator(log_file,
                                                size_guidance={ # see below regarding this argument
                                                  event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                  event_accumulator.IMAGES: 4,
                                                  event_accumulator.AUDIO: 4,
                                                  event_accumulator.SCALARS: 0,
                                                  event_accumulator.HISTOGRAMS: 1,
                                              })

        ea.Reload() # loads events from file

        if save_best == 'metric':
            scalar_name = metric + '_val'
            score = [s.value for s in ea.Scalars(scalar_name)]
            best_score = max(score)

        elif save_best == 'loss':
            scalar_name = 'Loss_val'
            score = [s.value for s in ea.Scalars(scalar_name)]
            best_score = min(score)

        next_epoch = len(score)

        print('Resuming training from epoch', next_epoch)
        print('The best', save_best, 'so far is', best_score)
        print()

    except:

        best_score = 0.0 if save_best == 'metric' else float("inf")

        next_epoch = 0

    return next_epoch, best_score


# # Train Function

# In[15]:


def train_model(parameters, model, model_name, loss_list, loss_name, dataloaders, criterion, optimizer, scheduler, is_inception=False):

    since = time.time()

    # Get State from Tensorborad Log
    log_dir = os.path.join(parameters['directory']['logs'], model_name)
    next_epoch, best_score = getLogData(log_dir, parameters['save_best'], parameters['metric'])

    # Start Tensorborad
    tensorboard = SummaryWriter(log_dir=log_dir)

    epoch_metric = 0.0

    print(model_name)
    print()

    is_cuda, gpu_list, device = getCudaDevices(parameters)
    model_dir = parameters['directory']['model']
    num_epochs=parameters['num_epoch']
    save_best=parameters['save_best']
    metric=parameters['metric']
    num_classes = parameters['num_classes']

    for epoch in range(next_epoch, num_epochs):

        print(' Epoch {}/{} '.format(epoch, num_epochs - 1).center(100, '='))

        epoch_since = time.time()
        lr = optimizer.param_groups[0]['lr']

        print('Learning Rate:', lr)
        tensorboard.add_scalar('LR', lr, epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_preds = []
            running_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)

                        loss1, preds = calcLoss(loss_list, criterion, loss_name, outputs, labels, num_classes)
                        loss2, preds = calcLoss(loss_list, criterion, loss_name, aux_outputs, labels, num_classes)

                        loss = loss1 + 0.4*loss2

                    else:

                        outputs = model(inputs)

                        outputs = outputs.squeeze()

                        loss, preds = calcLoss(loss_list, criterion, loss_name, outputs, labels, num_classes)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                running_preds = np.append(running_preds, preds.squeeze().cpu().detach().numpy())
                running_labels = np.append(running_labels, labels.squeeze().cpu().detach().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if metric:
                epoch_metric = calcMetric(running_preds, running_labels, metric)
                tensorboard.add_scalar('{} {}'.format(metric, phase), epoch_metric, epoch)

            print('{} Loss: {:.4f} {}: {:.4f}'.format(phase, epoch_loss, metric, epoch_metric))

            # Save the best model
            if phase == 'val':

                if scheduler:
                    scheduler.step(epoch_loss)

                save_flag = False

                if save_best == 'metric' and epoch_metric > best_score:

                    best_score = epoch_metric
                    save_flag = True

                elif save_best == 'loss' and epoch_loss < best_score:

                    best_score = epoch_loss
                    save_flag = True

                if save_flag:
                    print('Saving the best model at {}'.format(model_dir))
                    torch.save(model.state_dict(), model_dir + '/' + model_name + '_' + save_best + str(best_score) + '.pt')

                epoch_time_elapsed = time.time() - epoch_since
                print('Epoch time elapsed: {:.0f}m {:.0f}s'.format(epoch_time_elapsed // 60, epoch_time_elapsed % 60))

            # Write loss into Tensorboard
            tensorboard.add_scalar('Loss {}'.format(phase), epoch_loss, epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val {}: {:4f}'.format(save_best, best_score))

    return best_score


# # Grid Search

# In[16]:

def GridSearch(net_list, optimizer_list, loss_list, scheduler_list, parameters):

    model_name_list = []
    metric_list = []

    for inp in parameters['input_sizes']:

        for frac in parameters['sample_fracs']:

            for bch in parameters['batch_sizes']:

                dataloaders_dict = getDataLoaders(inp, frac, bch, parameters)
                augmentation_tag = parameters['data_augmentation']['tag']

                for m in parameters['models']:

                    model_parameters = net_list[m]
                    base_model = model_parameters['base_model']
                    model = getModel(net_list, m, parameters)

                    for o in parameters['optimizers']:

                        optimizer = getOptimizer(optimizer_list, o, model)

                        for s in parameters['schedulers']:

                            scheduler = getScheduler(scheduler_list, s, optimizer)

                            for l in parameters['losses']:

                                criterion = getLossFunction(loss_list, l)

                                model_name = f'{base_model}_Inp{str(inp)}-{augmentation_tag}-Data{str(frac)}-Bch{str(bch)}-{m}-{s}-{o}-{l}'

                                #summary(model, input_size=(CHANNELS, inp, inp))

                                # Train and evaluate
                                best_score = train_model(
                                    parameters,
                                    model,
                                    model_name,
                                    loss_list,
                                    l,
                                    dataloaders_dict,
                                    criterion,
                                    optimizer,
                                    scheduler,
                                    is_inception=net_list[m]['is_inception'])

                                model_name_list.append(model_name)
                                metric_list.append(best_score)


# # The Best Model Metrics

# In[ ]:


#fig, ax = plt.subplots()
#width = 0.75 # the width of the bars
#ind = np.arange(len(metric_list))  # the x locations for the groups
#ax.barh(ind, metric_list, width)
#ax.set_yticks(ind+width/2)
#ax.set_yticklabels(model_name_list, minor=False)
#plt.xlabel('Loss')
#for i, v in enumerate(metric_list):
#    ax.text(v, i, str(v))

def main():

    # Print Title
    print(" Auto Deep Learning ".center(100, '='))

    # Get Parameters
    with open('core/net_list.json') as f:
        NET_LIST = json.load(f)

    with open('core/loss_list.json') as f:
        LOSS_LIST = json.load(f)

    with open('core/optimizer_list.json') as f:
        OPTIMIZER_LIST = json.load(f)

    with open('core/scheduler_list.json') as f:
        SCHEDULER_LIST = json.load(f)

    with open('parameters/diabetic_retinopathy.json') as f:
        parameters = json.load(f)

    PrintCombinations(parameters)

    is_cuda, gpu_list, device = getCudaDevices(parameters)

    # # Calc Classes Weight
    if parameters['num_classes'] > 1:

        distrib_freq = train_dataset.y.sum().to_numpy()

        w_classes = distrib_freq.sum() / (parameters['num_classes'] * distrib_freq)

        for l in parameters['losses']:
            if 'weight' in LOSS_LIST[l]:
                LOSS_LIST[l]['weight'] = torch.from_numpy(w_classes).to(device)

    GridSearch(NET_LIST, OPTIMIZER_LIST, LOSS_LIST, SCHEDULER_LIST, parameters)

if __name__ == '__main__':
    main()
