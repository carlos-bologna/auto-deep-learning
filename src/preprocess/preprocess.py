import random
import cv2
from PIL import Image
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import concurrent.futures
from sklearn.cluster import KMeans
from collections import Counter

kmeans = None

def CropROI(img):
    global kmeans
    x = img[img.shape[0] // 2, : , :].sum(1)
    xk = kmeans.predict(x.reshape(-1, 1))
    
    # Make sure that monority classe is zero
    freq_distrib = Counter(xk)
    if freq_distrib[0] > freq_distrib[1]:
        xk = 1 - xk
    
    x1 = np.argmax(xk)
    x2 = img.shape[1] - np.argmax(np.flip(xk, axis=0))
    
    y = img[:, img.shape[1] // 2 , :].sum(1)
    yk = kmeans.predict(y.reshape(-1, 1))
    y1 = np.argmax(yk)
    y2 = img.shape[0] - np.argmax(np.flip(yk, axis=0))
    
    return img[y1:y2, x1:x2, : ]

def makeSquared(img):
    h, w = img.shape[0], img.shape[1]
    
    if h == w:
        return img
    
    margim = abs(h - w) // 2
    
    if h < w:
        img = img[:, margim:margim + h]
    else:
        img = img[margim:margim + w, :]
    
    return img

def normalize(img):
    return (img - MEAN) / STD

def resize_and_convert(files):

    # Source File
    src_file = files[0]
    dst_file = files[1]
    
    if os.path.exists(dst_file):
        return
    
    # Read Image as BGR colors layers
    img = np.array(Image.open(src_file))

    # Convert from BGR to RGB
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop Useless Back Margim
    img = CropROI(img)
    
    # Make Image Squared
    img = makeSquared(img)
    
    # Resize
    target_size = dst_file.split('/')[-3]
    img = cv2.resize(img, (int(target_size), int(target_size)))
    
    # Normalize
    #img = normalize(img)

    # Save
    np.save(dst_file, img.astype(np.uint8))
    
        
def Preprocess(data_dir, x, y, input_size, clear_cache, dst_dir):

    global kmeans
    
    # Get all source files
    files = list(map(lambda r: r[0] + '/' + r[1], zip(y, x)))
    
    # Pick one files
    rdn_file = random.choice(files)
    
    # Read files
    img = cv2.imread(os.path.join(data_dir, rdn_file))

    # If image is too big, resize it.
    if img.shape[0] > 2000 or img.shape[1] > 2000:
        img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))

    # Cluster File Values
    kmeans = KMeans(n_clusters=2, random_state=0).fit(img.flatten().reshape(-1, 1))

    # Clear All Previous Preprocess Images
    if clear_cache:
        shutil.rmtree(os.path.join(dst_dir, str(input_size)), ignore_errors=True)
    
    # Check/Create Destination Directory
    for d in set(y):
        if not os.path.isdir(os.path.join(dst_dir, str(input_size), str(d))):
            os.makedirs(os.path.join(dst_dir, str(input_size), str(d)))
    
    # Make Source Full Path
    src_files = list(map(lambda x: data_dir + '/' + x , files))
    dst_files = list(map(lambda x: dst_dir + '/' + str(input_size) + '/' + x.split('.')[0] + '.npy' , files))
    path_list = list(zip(src_files, dst_files))
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        for result in executor.map(resize_and_convert, path_list):
            pass