#!/usr/bin/env python
# coding: utf-8

# In[107]:


import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# In[108]:


# pip install keras


# In[109]:


from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
import cv2


# In[110]:


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


# In[111]:

"""
train_path = '/home/vraskap3/datas/train/data/'#enter path to training data
test_path = '/home/vraskap3/datas/test/data/'#enter path to testing data
mask_path = '/home/vraskap3/datas/masks/'
output_path = '/home/vraskap3/datas/output/'
"""
train_path = 'C:\\Users\\Nathan\\vraska-p3\\data\\processed\\data\\'#enter path to training data
#test_path = '/data/processed/data'#enter path to testing data
mask_path = 'C:\\Users\\Nathan\\vraska-p3\\data\\processed\\masks\\'
output_path = 'C:\\Users\\Nathan\\vraska-p3\\data\\output\\'

# In[112]:


train_ids = next(os.walk(mask_path))[2]
train_ids = [x[:-4] for x in train_ids]
# print(train_ids)
# exit()
#test_ids = next(os.walk(test_path))[1]
# print(test_ids)


# In[113]:


X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)


# In[114]:


print("X_train",X_train.shape)
print("Y_train",Y_train.shape)


# In[115]:


print('Getting and resizing train images and masks ... ')
sys.stdout.flush()


# In[101]:

"""
for n, ids in tqdm(enumerate(train_ids), total=len(train_ids)):
#     print("value of n" , n)
#     print("value of id" , ids)
    path = train_path + ids +'/'
#     print(path)
    image = imread(path +'frame0000.png')[:,:]
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
#     print(image)
    X_train[n] = image
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#     print(mask)
    mask_one = imread(mask_path + ids + '.png')[:,:]
    mask = np.where(mask_one == 1, 0, mask_one)
#     mask = np.where(mask == 2, 1, mask)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), preserve_range=True)
#     mask = np.maximum(mask, mask_one)
    
#     print(mask)
    Y_train[n] = mask

"""
# In[28]:


# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# model.summary()


# In[26]:


callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


# In[ ]:


#results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=10, epochs=1, )
#                     callbacks=callbacks)


# In[ ]:


# X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
# sizes_test = []
# sys.stdout.flush()
# for n,ids in tqdm(enumerate(test_ids), total=len(test_ids)):
#     path = test_path + ids +'/'
#     img = imread(path +'frame0000.png')[:,:]
#     sizes_test.append([img.shape[0], img.shape[1]])
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
# #     print(ids)
#     X_test[n] = img
# #     print(type(X_test[n]))


# In[106]:


#model.load_weights('model-dsbowl2018-1.h5')


# In[ ]:


# for x in test_ids:
#     path = test_path + x +'/'
#     img =imread(path +'frame0000.png')[:,:]
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
# #     img=np.expand_dims(img,axis=-1)
#     X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=int)
#     X_test[0] = img
#     predictions = model.predict(X_test)
#     mask=predictions[0]
#     print(mask)
# #     print(type(mask))
    
#     img_name = output_path + x + ".png"
#     cv2.imwrite(img_name,mask)


# In[12]:


# img_name_prefix = "frame00"
# img_name_suffix = ".png"
# image_names = []
# img_id_numbers = np.arange(0, 10, 1)
# for i in range(len(img_id_numbers)):
#     img_id = f'0{img_id_numbers[i]}' if img_id_numbers[i] < 10 else f'{img_id_numbers[i]}'
#     image_names.append(img_name_prefix + img_id + img_name_suffix)

# image_names


# In[24]:


# for image_name in image_names:
#     print(f'going over the frame {image_name}')
#     for n, ids in tqdm(enumerate(train_ids), total=len(train_ids)):
#     #     print("value of n" , n)
#     #     print("value of id" , ids)
#         path = train_path + ids +'/'
# #     print(path)
    
#         image = imread(path + image_name)[:,:]
#         image = resize(image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
#     #     print(image)
#         X_train[n] = image
#         mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#     #     print(mask)
#         mask_one = imread(mask_path + ids + '.png')[:,:]
#         mask = np.where(mask_one == 1, 0, mask_one)
#     #     mask = np.where(mask == 2, 1, mask)
#         mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), preserve_range=True)
#     #     mask = np.maximum(mask, mask_one)

#     #     print(mask)
#         Y_train[n] = mask
#     results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=10, epochs=1, )
# #                 callbacks=callbacks)


# In[17]:


# path = test_path + '0193e929dfdc1a9854ceac030a1339bb75f2f7cde153deeded176f4e38be39bd' +'/'
# img =imread(path +'frame0020.png')[:,:]
# img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
# #     img=np.expand_dims(img,axis=-1)
# X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=int)
# X_test[0] = img
# predictions = model.predict(X_test)
# mask=predictions[0]
# mask = np.where(mask > 0.5, 1, 0)
# np.set_printoptions(threshold=sys.maxsize)
# print(mask.squeeze())
# #   


# In[116]:


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


# In[117]:


# @tf.function
def load_image_train(image, mask):
    input_image = tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    input_mask = tf.image.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# In[118]:


def load_image_test(image, mask):
    input_image = tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    input_mask = tf.image.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# In[119]:


def display(display_list):
    plt.figure(figsize=(40, 40))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        print(i)
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# In[ ]:


# for image, mask in train.take(1):
#     sample_image, sample_mask = image, mask
# display([sample_image, sample_mask])


# In[30]:


# for n, ids in tqdm(enumerate(train_ids), total=len(train_ids)):
#     print("value of n" , n)
#     print("value of id" , ids)


# ids = "007f736aedbc4ca67989f8ca62f1bbeb447ad76698351fe387923963ee50e5ae"
# path = train_path + ids +'/'
# #     print(path)
# image = imread(path +'frame0000.png')[:,:]
# image = resize(image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)

# mask_one = imread(mask_path + ids + '.png')[:,:]
# mask = resize(mask_one, (IMG_HEIGHT, IMG_WIDTH, 1), preserve_range=True)

# in_image, in_mask = load_image_train(image, mask)
# in_image


# #     print(image)
# X_train[n] = image
# mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
# #     print(mask)
# mask_one = imread(mask_path + ids + '.png')[:,:]
# mask = np.where(mask_one == 1, 0, mask_one)
# #     mask = np.where(mask == 2, 1, mask)
# #     mask = np.maximum(mask, mask_one)

# #     print(mask)
# Y_train[n] = mask


# In[120]:


#display([in_image])


# In[121]:


#tf.keras.preprocessing.image.array_to_img(in_image)


# In[122]:


#tf.keras.preprocessing.image.array_to_img(in_mask)


# In[123]:


base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_WIDTH, IMG_HEIGHT, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False


# In[124]:


# !pip install -q git+https://github.com/tensorflow/examples.git


# In[125]:


from tensorflow_examples.models.pix2pix import pix2pix

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


# In[126]:


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3])

  # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

  # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# In[127]:


OUTPUT_CHANNELS = 3


# In[154]:


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[155]:


model.summary()


# In[ ]:





# In[156]:


tf_tensors_images = []
tf_tensors_masks = []
for n, ids in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = train_path + ids +'/'
    image = imread(path +'frame0000.png')[:,:]
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)

    mask_one = imread(mask_path + ids + '.png')[:,:]
    mask = resize(mask_one, (IMG_HEIGHT, IMG_WIDTH, 1), preserve_range=True)

    in_image, in_mask = load_image_train(image, mask)
    tf_tensors_images.append(in_image)
    tf_tensors_masks.append(in_mask)


# In[157]:


#tf.keras.preprocessing.image.array_to_img(tf_tensors_images[0])


# In[158]:


#tf.keras.preprocessing.image.array_to_img(tf_tensors_masks[0])


# In[159]:


dataset = tf.data.Dataset.from_tensor_slices((tf_tensors_images, tf_tensors_masks))


# In[160]:


#dataset


# In[161]:

BUFFER_SIZE = 128
BATCH_SIZE = 16
train = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# In[162]:


print(train)


# In[ ]:


history = model.fit(train, epochs=1)
