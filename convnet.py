# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import cntk as C





# Create the binary convolution network for training.
def create_binary_convolution_model():

    # Input variables denoting the features and label data
    feature_var = C.input((num_channels, image_height, image_width))
    label_var = C.input((num_classes))

    # apply model to input
    scaled_input = C.element_times(C.constant(0.00390625), feature_var)

    # first layer is ok to be full precision
    z = C.layers.Convolution((3, 3), 32, pad=True, activation=C.relu)(scaled_input)
    z = C.layers.MaxPooling((3,3), strides=(2,2))(z)

    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = C.layers.Convolution((3,3), 128, pad=True)(z)
    z = C.layers.MaxPooling((3,3), strides=(2,2))(z)

    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = C.layers.Convolution((3,3), 128,  pad=True)(z)
    z = C.layers.MaxPooling((3,3), strides=(2,2))(z)

    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = C.layers.Convolution((1,1), num_classes,  pad=True)(z)
    z = C.layers.AveragePooling((z.shape[1], z.shape[2]))(z)
    z = C.reshape(z, (num_classes,))


    # After the last layer, we need to apply a learnable scale
    SP = C.parameter(shape=z.shape, init=0.001)
    z = C.element_times(z, SP)
    z = C.softmax(z)
    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    return C.combine([z, ce, pe])

# Clones a binary convolution network, sharing the original parameters  but substitutes the
# python 'binary_convolve' Function instances used during training, faster C++ NativeBinaryConvolveFunction
# instances that uses optimized binary convolution implementations generated using the Halide framework


def get_z_and_criterion(combined_model):
    return (C.combine([combined_model.outputs[0].owner]), C.combine([combined_model.outputs[1].owner, combined_model.outputs[2].owner]))

# Import training and evaluation routines from ConvNet_CIFAR10_DataAug
abs_path = os.path.dirname(os.path.abspath(__file__))
custom_convolution_ops_dir = os.path.join(abs_path, "..", "..", "Image", "Classification", "ConvNet", "Python")
sys.path.append(custom_convolution_ops_dir)

from ConvNet_CIFAR10_DataAug import *

############################# 
# main function boilerplate #
#############################

if __name__=='__main__':
    model = create_binary_convolution_model()
    z, criterion = get_z_and_criterion(model)
    reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
    train_model(reader_train, z, criterion, max_epochs=200)


    model_path = data_path + "/model.cntk"
    model.save(model_path)

    
    reader_test = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)


    evaluate(reader_test, criterion, minibatch_size=1, max_samples=1000)
