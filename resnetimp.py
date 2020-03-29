from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Input
from keras import layers

from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape 
from keras.engine.topology import get_source_inputs

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

#now making the identity block

def identity_block(input_tensor,kernel_size,filters,stage,block):
    filter1,filter2,filter3 = filters
    
    if K.image_data_format == 'channel_last':
        bn_axis=3
    else:
        bn_axis=1
    
    convname_base = 'res' + str(stage) + block + '_branch'
    bnname_base = 'bn' + str(stage) + block + '_branch'
    
    X=Conv2D( filter1,(1,1),name= convname_base +'2a')(input_tensor)
    X=BatchNormalization(axis=bn_axis, name=bnname_base + '2a')(X)
    X=Activation('relu')(X)
    
    X = Conv2D(filter2, kernel_size,
               padding='same', name=convname_base + '2b')(X)
    X = BatchNormalization(axis=bn_axis, name=bnname_base + '2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filter3, (1, 1), name=convname_base + '2c')(X)
    X = BatchNormalization(axis=bn_axis, name=bnname_base + '2c')(X)

    X = layers.add([X, input_tensor])
    X = Activation('relu')(X)
    return X

#now lets implement the convolution block which will include our skip connection

def convolution_block(input_tensor,kernel_size,filters,stage,block,strides=(2, 2)):
    
    filter1,filter2,filter3 = filters
    
    if K.image_data_format == 'channel_last':
        bn_axis=3
    else:
        bn_axis=1
    
    convname_base = 'res' + str(stage) + block + '_branch'
    bnname_base = 'bn' + str(stage) + block + '_branch'
    
    X=Conv2D( filter1,(1,1),strides=strides,name= convname_base +'2a')(input_tensor)
    X=BatchNormalization(axis=bn_axis, name=bnname_base + '2a')(X)
    X=Activation('relu')(X)
    
    X = Conv2D(filter2, kernel_size,
               padding='same', name=convname_base + '2b')(X)
    X = BatchNormalization(axis=bn_axis, name=bnname_base + '2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filter3, (1, 1), name=convname_base + '2c')(X)
    X = BatchNormalization(axis=bn_axis, name=bnname_base + '2c')(X)

    #now the shortcut block
    shortcut = Conv2D(filter3,(1,1),strides=strides,name=convname_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name= bnname_base + '1')(shortcut)
    
    X= layers.add([X,shortcut])
    X=Activation('relu')(X)
    return X

#in this implementation we will be implementing the resnet50 architecture

def Resnet(include_top=True,input_tensor = None,input_shape=None,weights='imagenet',pooling=None,classes=1000):
    #include_top = whether to include fully connected layer at the top of network
    
    if weights not in {'imagenet', None}:
        raise ValueError('the weight should be initialized from imagenet')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    
    input_shape = _obtain_input_shape(input_shape,default_size=224,min_size=197,
                                      data_format=K.image_data_format,
                                      require_flatten=include_top)
    
    if input_tensor is None:
        input_img = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            input_img= Input(shape=input_shape,tensor=input_tensor)
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    #now we will be building the resnet50 architecture 
    #our CNN will be in the following pattern
    # padding -> convolutional layer -> batchnorm -> activation ->max pooling 
    X = ZeroPadding2D((3, 3))(input_img)
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=bn_axis, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    print(X.shape)
    X = convolution_block(X, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    print(X.shape)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    print(X.shape)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    print(X.shape)


    X = convolution_block(X, 3, [128, 128, 512], stage=3, block='a')
    print(X.shape)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    print(X.shape)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    print(X.shape)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    print(X.shape)

    X = convolution_block(X, 3, [256, 256, 1024], stage=4, block='a')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolution_block(X, 3, [512, 512, 2048], stage=5, block='a')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    
    #according to the Deep Residual Learning for Image Recognition paper,average pooling has been used for this architecture
    X=AveragePooling2D((7,7),name='averagepool')(X)
    
    if include_top:
        X=Flatten()(X)
        X-Dense(classes,activation='softmax',name="fc1000")(X)
    
    else:
        if pooling == 'avg':
            X=GlobalAveragePooling2D()(X)
        elif pooling == "max":
            X=GlobalMaxPooling2D()(X)
            
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = input_img
    
    model = Model(inputs,X,name='resnet50')
    
    if weights == 'imagenet':
        if include_top:
                        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
                        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
    
    return model


if __name__ == '__main__':
    model = Resnet(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    X = preprocess_input(X)
    print('Input image shape:', X.shape)

    preds = model.predict(X)
    print('Predicted:', decode_predictions(preds))

