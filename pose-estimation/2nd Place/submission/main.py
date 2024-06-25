import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
# import tifffile
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import gc
import warnings
warnings.filterwarnings("ignore")

#import random
# import torch
# import torch.nn as nn
#import albumentations

# import segmentation_models_pytorch as smp
#import skimage


#from tqdm import tqdm
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader,Dataset
# from fastprogress import master_bar, progress_bar


CENTER_CROP=384
RESIZE=384
NUM_TARGETS=1

#import dill as pickle
from ultralytics import YOLO
detection_model = YOLO('assets/last.pt')
#import torch
#detection_model = torch.hub.load('.', 'custom', path='assets/last.pt', source='local') 
#detection_model = YOLO('yolov8n.yaml').load('assets/last.pt')
#detection_model = YOLO('assets/best.onnx')

def detect_center(model, impath):
    center=(640, 512)
    try:
#        results = model(impath, verbose=False)
        results = model.predict(impath, verbose=False)
    
        for result in results:
            for box in result.boxes:
                # print(box)
                left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=np.int64).squeeze()
#                width = right - left
#                height = bottom - top
                center = (left + int((right-left)/2), top + int((bottom-top)/2))
                # print(center)
    except:
        center=(640, 512)
        
    return center
#center = detect_center(detection_model, impath)

def sat_center_crop(img, center=(512, 640), SIZE=RESIZE):
    y, x = img.shape[:2]
    halfS=int(SIZE/2)
    y1 = min(y - SIZE, max(0, center[0]-halfS))
    y2 = min(y, max(SIZE, center[0]+halfS))
    x1 = min(x - SIZE, max(0, center[1]-halfS))
    x2 = min(x, max(SIZE, center[1]+halfS))
#     print(y2,x2)
#     print(y1,y2,x1,x2)
    return img[y1:y2,x1:x2,:]

def create_dir(path):
    if os.path.isdir(path)==False:
        os.makedirs(path)

def np_power_with_sign(ar, power=0.999):
    signs = np.sign(ar)
    return ((np.abs(ar))**power)*signs

def center_crop(img, final_size=CENTER_CROP):
    y, x = img.shape[:2]
    y2 = max(0, int(y/2-final_size/2))
    x2 = max(0, int(x/2-final_size/2))
#     print(y2,x2)
    return img[y2:-y2,x2:-x2,:]

def np_log_with_sign(ar):
    signs = np.sign(ar)
    return np.log(np.abs(ar))*signs

def np_exp_with_sign(ar):
    signs = np.sign(ar)
    return np.exp(np.abs(ar))*signs
    
#name='siam_v2'
#weightsname=name+'.hdf5' 
#weightsname='ep39siam_v19b.hdf5' 
weightsname='ep79siam_cc_v19B2xb.hdf5' 

data_path='/code_execution/data/'


# train_labels=pd.read_csv('train_labels.csv')
#rangeDF=pd.read_csv('range.csv')





#from tensorflow.keras.applications import  EfficientNetB1
#import keras
from tensorflow.keras.applications import  EfficientNetB0

#from keras import regularizers
from keras import backend as K
# from keras.preprocessing.image import img_to_array,array_to_img

from keras.optimizers import Adam
from keras.layers import Input
from keras.layers import Concatenate, Dense, Flatten 
#from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten 
from keras.layers import GlobalAveragePooling2D, Lambda, Dropout
#from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape, Dropout
from keras.models import Model
#import tensorflow.keras.layers as L

#from keras.applications.densenet import DenseNet201, preprocess_input



#img_shape = img.shape#(input_size, input_size, 3)
img_shape=(RESIZE,RESIZE,3)
# regul  = regularizers.l2(0.0002)

#def subblock(x, filter, **kwargs):
#    x = BatchNormalization()(x)
#    y = x
#    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y) # Reduce the number of features to 'filter'
#    y = BatchNormalization()(y)
#    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y) # Extend the feature field
#    y = BatchNormalization()(y)
#    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y) # no activation # Restore the number of original features
#    y = Add()([x,y]) # Add the bypass connection
#    y = Activation('relu')(y)
#    return y

def build_model(img_shape=(512,512,3), freeze_backbone=False, do=0.2):

    
    backbone = EfficientNetB0(include_top=False,  weights=None,  input_tensor=None,  input_shape=img_shape,  pooling=None)#     classes=1000,#     classifier_activation="softmax",#     **kwargs)
    
    output = GlobalAveragePooling2D()(backbone.output) 
#     output = GlobalMaxPooling2D()(backbone.output) 
    branch_model = Model(backbone.input,output)
    if freeze_backbone:
        for layer in branch_model.layers:
            layer.trainable = False

    del backbone

    ############
    # HEAD MODEL
    ############
#     mid        = 32
    xa_inp     = Input(shape=branch_model.output_shape[1:])
    xb_inp     = Input(shape=branch_model.output_shape[1:])

    x1         = Lambda(lambda x : x[0] - x[1])([xa_inp, xb_inp])
    x2         = Lambda(lambda x : K.square(x), output_shape=x1.shape[1:])(x1)
    x          = Concatenate()([x1, x2])
    x          = Flatten(name='flatten')(x)

    x         = Dropout(0.2)(x)
    x         = Dense(1024, activation='relu')(x)
    x         = Dropout(0.1)(x)
    x         = Dense(512, activation='relu')(x)
    # Weighted sum implemented as a Dense layer.
    out          = Dense(NUM_TARGETS, activation='linear', name='weighted-average')(x)

    head_model = Model([xa_inp, xb_inp], out, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a      = Input(shape=img_shape)
    img_b      = Input(shape=img_shape)
    xa         = branch_model(img_a)
    xb         = branch_model(img_b)
    y          = head_model([xa, xb])   

    model      = Model([img_a, img_b], y)
    
    return model, branch_model, head_model

model, branch_model, head_model = build_model(img_shape=(RESIZE,RESIZE,3), freeze_backbone=True)
#head_model.summary()



#cases = os.listdir(data_path)
cases = os.listdir(data_path+'images')
#len(cases)


val_ids=[]
val_targets=[]
for c in cases:
#    p=data_path+c+'/'
    p=data_path+'images/'+c+'/'
    all_images=os.listdir(p)
    all_images=[x for x in all_images if x not in ['000.png']]
    all_images.sort()
#    print(all_images)
    for im in all_images:
        val_ids.append([p+'000.png', p+im])
        val_targets.append(np.zeros(7))
    
val_ids=np.vstack(val_ids)
val_targets=np.vstack(val_targets)    
#len(val_ids),val_targets.shape



val_chains=[x[1].split('/')[-2] for x in val_ids]
val_chains_i=[int(x[1].split('/')[-1][:-4]) for x in val_ids]
ref=pd.DataFrame({'chain_id': np.unique(val_chains)})
ref['i']=0
ref['x']=0
ref['y']=0
ref['z']=0
ref['qw']=1
ref['qx']=0
ref['qy']=0
ref['qz']=0
# train_labels



#val_shapes=[]
#val_failed_reading=[]
#for c in cases:
##    p=data_path+c+'/'
#    p=data_path+'images/'+c+'/'
#    all_images=os.listdir(p)
#    all_images.sort()
##     all_images=[x for x in all_images if x not in ['000.png']]
#    for im in all_images:
#        try:
#            img = cv2.imread(p+im)
#            val_shapes.append( [p+im, img.shape])
#        except:
#            val_failed_reading.append( p+im)
            
# (array([(1024, 1280, 3)], dtype=object),
#  ['/fast/Spacecraft_Pose_Estimation/images/898466710c/059.png',
#   '/fast/Spacecraft_Pose_Estimation/images/575ebc7410/040.png',
#   '/fast/Spacecraft_Pose_Estimation/images/af0973ade8/059.png'])    
# for n in val_failed_reading:
#     print(np.where(val_ids[:,1]==n)[0][0])
# 856
# 1285
# 1747

#
#for n in val_failed_reading:
#    print(np.where(val_ids[:,1]==n)[0][0])

#
#val_ids=val_ids[[x for x in range(len(val_ids)) if x not in [856, 1285, 1747]],:]
#val_targets=val_targets[[x for x in range(len(val_targets)) if x not in [856, 1285, 1747]],:]









#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#import tensorflow as tf

#weightsname=name+'.hdf5' 
model.compile(Adam(lr=0.0001), loss='mse')
# bs=0
# bc=0
# zerolabelnum=1
# batch_size_multiplier = int(16/(zerolabelnum+1))
batch_size = 1
#
model.load_weights('assets/'+weightsname)
    




#val_preds=model.predict( data_generator_siam(val_ids, val_targets), 
#                        steps=np.ceil(len(val_ids)/batch_size),
#                       batch_size=batch_size)
#
#def data_generator_siam_1(ids, labels):
#
#    x1_batch = []
#    x2_batch = []
#    y_batch = labels.copy()
#
##    y_batch[:3] = np_log_with_sign(y_batch[:3])
#
#    img1 = cv2.imread(ids[0])
#    img1 = center_crop(img1)
##                 img1 = cv2.resize(img1, (RESIZE, RESIZE), cv2.INTER_AREA)
#    img2 = cv2.imread(ids[1])
#    img2 = center_crop(img2)
##                 img2 = cv2.resize(img2, (RESIZE, RESIZE), cv2.INTER_AREA)                
#    x1_batch.append(img1)
#    x2_batch.append(img2)    
#
#    x1_batch = np.array(x1_batch, np.float32) #/ 255
##             x1_batch=x1_batch[...,::-1] # BGR to RGB   #for DENSENET
#    x2_batch = np.array(x2_batch, np.float32) #/ 255
##     print(x1_batch.shape, y_batch.shape)
#
#    yield [x1_batch.astype('uint8'), x2_batch.astype('uint8')], y_batch 

def data_generator_siam_1(ids, labels):

    try:
        centerW0, centerH0 = detect_center(detection_model, ids[0])
    except:
        centerW0, centerH0 = 640, 512
    try:
        centerW1, centerH1 = detect_center(detection_model, ids[1])
    except:
        centerW1, centerH1 = 640, 512
    
    x1_batch = []
    x2_batch = []
    y_batch = labels.copy()

#    y_batch[:3] = np_log_with_sign(y_batch[:3])

    img1 = cv2.imread(ids[0])
    img1 = sat_center_crop(img1, center=( centerH0, centerW0))
#    img1 = center_crop(img1)
#                 img1 = cv2.resize(img1, (RESIZE, RESIZE), cv2.INTER_AREA)
    img2 = cv2.imread(ids[1])
    img2 = sat_center_crop(img2, center=( centerH1, centerW1))
#    img2 = center_crop(img2)
#                 img2 = cv2.resize(img2, (RESIZE, RESIZE), cv2.INTER_AREA)                
    x1_batch.append(img1)
    x2_batch.append(img2)    

    x1_batch = np.array(x1_batch, np.float32) #/ 255
#             x1_batch=x1_batch[...,::-1] # BGR to RGB   #for DENSENET
    x2_batch = np.array(x2_batch, np.float32) #/ 255
#     print(x1_batch.shape, y_batch.shape)

    yield [x1_batch.astype('uint8'), x2_batch.astype('uint8')], y_batch     


batch_size=1
val_preds=[]
sat0='init'
pred0=0
for i in range(len(val_ids)):    
    pred=model.predict( data_generator_siam_1(val_ids[i], val_targets[i]), 
                                steps=1,
                                batch_size=1)
    # smooth predictions within same satellite (post process)
    sat = val_ids[i][0].split('/')[-2]
    if sat==sat0: # same satellite
        pred= 0.9*pred + 0.1*pred0
        pred0=pred
    else:
        sat0=sat
        pred0=pred
        
    val_preds.append(pred)
#for i in range(len(val_ids)):
#    val_preds.append(model.predict( data_generator_siam_1(val_ids[i], val_targets[i]), 
#                            steps=1,
#                            batch_size=1,
##                            workers=1, use_multiprocessing=False
#)    )
val_preds = np.vstack(val_preds)
    

#val_predsD=val_preds.copy()
val_predsD=np.concatenate((val_preds,np.zeros((len(val_preds),6))),1)#

#     val_predsD[:,0] = np_exp_with_sign(val_predsD[:,0])
val_predsD[:,0] = np.clip(val_predsD[:,0], 0, 400) ** 1.02
#val_predsD[:,0] = np.clip(val_predsD[:,0], -10, 600) 
#val_predsD[:,0] = np_power_with_sign(np.clip(val_predsD[:,0], -10, 600) )
#val_predsD[:,1:3] = np.clip(val_predsD[:,1:3], 0, 0)

val_predsD[:,3:] = [1,0,0,0]

#val_predsD[:,:3] = np_exp_with_sign(val_predsD[:,:3])
#val_predsD[:,0] = np.clip(val_predsD[:,0], 0, 70)
#val_predsD[:,1:3] = np.clip(val_predsD[:,1:3], -1, 1)
## val_predsD[:,:3] = np.clip(np_exp_with_sign(val_predsD[:,:3]), -20, 20)

#ttm = np.array([ 1.90300843e+02, -3.87845548e+00, -3.04387253e+00,  3.52331717e-01,
#       -1.72995011e-01, -7.16987227e-02, -1.56063472e-01])
#val_predsD[:,3:] = [1,0,0,0] #(val_predsD[:,3:]+ttm[3:])/2
##val_predsD[:,3:] = (val_predsD[:,3:]+train_targets[:,3:].mean(0))/2

val_chain_preds_df = pd.DataFrame(np.concatenate((np.expand_dims(val_chains,-1), np.expand_dims(val_chains_i,-1), val_predsD),-1))
val_chain_preds_df.columns = ['chain_id', 'i', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'] # train_labels.columns
val_chain_preds_df.iloc[:,1:]=val_chain_preds_df.iloc[:,1:].astype(float)
val_chain_preds_df = pd.concat([val_chain_preds_df, ref]).sort_values(['chain_id','i']).reset_index(drop=True)




val_chain_preds_df.to_csv('/code_execution/submission/submission.csv',index=False)

