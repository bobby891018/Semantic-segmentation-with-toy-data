import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from tensorflow.keras import initializers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy import signal

def gauss_map(size_x, size_y=None, sigma_x=5, sigma_y=None):
    """
    create a 2d gaussian with certain sigma
    ----------
    size_x: image size in x-axis
    size_y: image size in y-axis
    sigma_x: standard deviation for gaussian in x-axis
    sigma_y: standard deviation for gaussian in y-axis
    ----------
    Returns
    - 2d gaussian array
    """
    if size_y == None:
        size_y = size_x
    if sigma_y == None:
        sigma_y = sigma_x
    
    assert isinstance(size_x, int)
    assert isinstance(size_y, int)
    
    x0 = size_x//2
    y0 = size_y//2
    
    x = np.arange(0, size_x, dtype=float)
    y = np.arange(0, size_y, dtype=float)[:,np.newaxis]
    
    x -= x0
    y -= y0
    
    exp_part = x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2)
    gaussian = 1/(2*np.pi*sigma_x*sigma_y)*np.exp(-exp_part)
    return gaussian/np.max(gaussian)

def mask_with_position(pos, flux, img_size, mask_value):
    """
    give a mask value in 2d array at certain position
    with fwhm in mask radius
    ----------
    pos: position of the sources
    flux: flux of the sources
    img_size: image size
    mask_value: the value that you want to insert in the array
    ----------
    Returns
    - masked array
    """
    mask = np.zeros((img_size, img_size))
    mask[pos] = flux
    mask = signal.convolve(mask, gaussian, mode='same')
    mask[mask<flux*0.5] = 0
    mask[mask>flux*0.5] = mask_value
    return mask

def blue_star(c1):
    """
    The relation between the color1 and color2 in BLUE stars
    assuming that the BLUE stars are 2 times brighter in color1
    ----------
    c1: flux in color1
    ----------
    Returns
    - flux in color2
    """
    c2 = 0.5*c1
    return c1, c2

def red_star(c2):
    """
    The relation between the color1 and color2 in RED stars
    assuming that the RED stars are 2 times brighter in color2
    ----------
    c2: flux in color2
    ----------
    Returns
    - flux in color1
    """
    c1 = 0.5*c2
    return c1, c2

''' make a gaussian kernel for the convolution
'''
gaussian = gauss_map(256, sigma_x=2)

''' generate our toy datasets
1/ create 50 images (256*256) with 10 blue stars and 10 red stars in each
2/ create the labeled mask regions with blue [1,0] and red [0,1] stars
3/ each stars have two colors in different channels
'''
def data_generator(data_size=50, img_size=256, source_number=10, tt_split=0.6):
    img = []
    for n in range(data_size):
        b_flux = blue_star(np.random.random(source_number))
        r_flux = red_star(np.random.random(source_number))
        b_pos = list(zip(random.sample(range(0, img_size-1), source_number),
                         random.sample(range(0, img_size-1), source_number)))
        r_pos = list(zip(random.sample(range(0, img_size-1), source_number),
                         random.sample(range(0, img_size-1), source_number)))

        img_c1 = 0.001*np.random.randn(img_size,img_size) # make noises in the map
        img_c2 = 0.001*np.random.randn(img_size,img_size)
        mask1 = np.zeros((img_size, img_size))
        mask2 = np.zeros((img_size, img_size))
        for s in range(source_number):
            img_c1[b_pos[s]] = b_flux[0][s]
            img_c2[b_pos[s]] = b_flux[1][s]
            img_c1[r_pos[s]] = r_flux[0][s]
            img_c2[r_pos[s]] = r_flux[1][s]
            
            mask1 += mask_with_position(b_pos[s], b_flux[0][s], img_size, 1)
            mask2 += mask_with_position(r_pos[s], r_flux[0][s], img_size, 1)
        img_c1 = signal.convolve(img_c1, gaussian, mode='same')
        img_c2 = signal.convolve(img_c2, gaussian, mode='same')
        mask = np.dstack((mask1, mask2))
        
        img_cn = np.dstack((img_c1, img_c2, mask))
        img.append(img_cn)
    img = np.array(img)

    train_x = img[0:int(data_size*tt_split),:,:,:-2]
    train_y = img[0:int(data_size*tt_split),:,:,-2:]
    test_x = img[int(data_size*tt_split):,:,:,:-2]
    test_y = img[int(data_size*tt_split):,:,:,-2:]
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = data_generator()

''' set some parameters for models
'''
shape = list(train_x[0].shape)
filters = [64, 32, 16, 8] 
kernels = len(filters)*[1]

''' create an Unet model to do the semantic segmantation
'''
class UNet(Model):
    def __init__(self, shape):
        self.shape = shape
        initializer = initializers.RandomUniform()
        activation = 'relu'
        
        inputs = Input(self.shape)
        conv1 = Conv2D(16, 3, activation=activation, padding='same', kernel_initializer=initializer)(inputs)
        conv1 = Conv2D(16, 3, activation=activation, padding='same', kernel_initializer=initializer)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(32, 3, activation=activation, padding='same', kernel_initializer=initializer)(pool1)
        conv2 = Conv2D(32, 3, activation=activation, padding='same', kernel_initializer=initializer)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer=initializer)(pool2)
        conv3 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer=initializer)(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer=initializer)(pool3)
        conv4 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer=initializer)(conv4)
        drop4 = Dropout(0.5)(conv4)

        up5 = Conv2D(64, 2, activation=activation, padding='same', kernel_initializer=initializer)(UpSampling2D(size = (2,2))(drop4))
        merge5 = tf.concat([drop3,up5], axis = 3)
        conv5 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer=initializer)(merge5)
        conv5 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer=initializer)(conv5)

        up6 = Conv2D(32, 2, activation=activation, padding='same', kernel_initializer=initializer)(UpSampling2D(size = (2,2))(conv5))
        merge6 = tf.concat([conv2,up6], axis = 3)
        conv6 = Conv2D(32, 3, activation=activation, padding='same', kernel_initializer=initializer)(merge6)
        conv6 = Conv2D(32, 3, activation=activation, padding='same', kernel_initializer=initializer)(conv6)

        up7 = Conv2D(16, 2, activation=activation, padding='same', kernel_initializer=initializer)(UpSampling2D(size = (2,2))(conv6))
        merge7 = tf.concat([conv1,up7], axis = 3)
        conv7 = Conv2D(16, 3, activation=activation, padding='same', kernel_initializer=initializer)(merge7)
        conv7 = Conv2D(16, 3, activation=activation, padding='same', kernel_initializer=initializer)(conv7)

        conv7 = BatchNormalization()(conv7)
        output = Conv2D(2, 1, activation='softmax', padding='same', kernel_initializer=initializer)(conv7)
        
        super(UNet, self).__init__(inputs=inputs, outputs=output)

'''train the models
'''
callbacks = EarlyStopping(monitor='loss', patience=5)
unet = UNet(shape)
unet.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy')
unet.fit(train_x, train_y, epochs=100, shuffle=True, callbacks=callbacks)

''' the predictions from test datasets
'''
img_pred = unet.predict(test_x)

def visualize(img_x, img_y, model, batch_no):
    """
    visualize the predictions from test datasets
    ----------
    img_x: x value from test datasets
    img_y: y value from test datasets
    model: trained model
    batch_no: batch number
    ----------
    Returns
    - visualizations of the predictions from test data
    """
    img_pred = model.predict(img_x)
    def plt_imshow(array, cmap='gray'):
        return plt.imshow(array, cmap=cmap, origin='lower')

    plt.subplot(2,2,1)
    plt.title("Mask (blue)")
    plt_imshow(img_y[batch_no,:,:,0])
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.title("Mask (red)")
    plt_imshow(img_y[batch_no,:,:,1])
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.title("predictions (blue)")
    plt_imshow(img_pred[batch_no,:,:,0])
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.title("predictions (red)")
    plt_imshow(img_pred[batch_no,:,:,1])
    plt.axis('off')

visualize(test_x, test_y, unet, 0)
plt.show()

plt.plot(test_y[0,:,:,0].flatten(), img_pred[0,:,:,0].flatten(), 'k.')
plt.show()


