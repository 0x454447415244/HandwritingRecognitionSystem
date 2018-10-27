###
# Copyright 2018 Edgard Chammas. All Rights Reserved.
# Licensed under the Creative Commons Attribution-NonCommercial International Public License, Version 4.0.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/legalcode
###

#!/usr/bin/python

import tensorflow as tf
import math

from config import cfg
from util import batch_norm_conv
from util import weight_variable
from util import conv2d
from util import max_pool

####################################################################
#CNN-specific architecture configuration
####################################################################
WND_HEIGHT = 64 		#Extraction window height
WND_WIDTH = 64			#Extraction window width
WND_SHIFT = WND_WIDTH - 2	#Window shift

MPoolLayers_ALL = 5	#Nbr of all maxpool layers
MPoolLayers_H = 2	#Nbr of maxpool in horizontal dimension
LastFilters = 512	#Nbr of feature maps at the last conv layer
####################################################################

FV = int(WND_HEIGHT / math.pow(2, MPoolLayers_ALL))

NFeatures = FV * LastFilters

def CNNLight(X, Training, Scope):

	with tf.variable_scope(Scope):

		ConvLayer1 = ConvLayer(X, 1, 64, Training, 'ConvLayer1')

		MPool1 = max_pool(ConvLayer1, ksize=(2, 2), stride=(2, 2))

		ConvLayer2 = ConvLayer(MPool1, 64, 128, Training, 'ConvLayer2')

		MPool2 = max_pool(ConvLayer2, ksize=(2, 2), stride=(2, 2))

		ConvLayer3 = ConvLayer(MPool2, 128, 256, Training, 'ConvLayer3')

		ConvLayer4 = ConvLayer(ConvLayer3, 256, 256, Training, 'ConvLayer4')

		MPool4 = max_pool(ConvLayer4, ksize=(2, 1), stride=(2, 1))

		ConvLayer5 = ConvLayer(MPool4, 256, 512, Training, 'ConvLayer5')

		ConvLayer6 = ConvLayer(ConvLayer5, 512, 512, Training, 'ConvLayer6')

		MPool6 = max_pool(ConvLayer6, ksize=(2, 1), stride=(2, 1))

		ConvLayer7 = ConvLayer(MPool6, 512, 512, Training, 'ConvLayer7')

		MPool7 = max_pool(ConvLayer7, ksize=(2, 1), stride=(2, 1))

		MPool7_T = tf.transpose(MPool7, perm=[0,2,1,3])

		MPool7_T_RSH = tf.reshape(MPool7_T, [-1, FV, LastFilters])

		return tf.reshape(MPool7_T_RSH, [-1, NFeatures])


def CNN(X, Training, Scope):

	with tf.variable_scope(Scope):

		ConvLayer1 = ConvLayer(X, 1, 64, Training, 'ConvLayer1')

		ConvLayer2 = ConvLayer(ConvLayer1, 64, 64, Training, 'ConvLayer2')

		MPool2 = max_pool(ConvLayer2, ksize=(2, 2), stride=(2, 2))

		ConvLayer3 = ConvLayer(MPool2, 64, 128, Training, 'ConvLayer3')

		ConvLayer4 = ConvLayer(ConvLayer3, 128, 128, Training, 'ConvLayer4')

		MPool4 = max_pool(ConvLayer4, ksize=(2, 2), stride=(2, 2))

		ConvLayer5 = ConvLayer(MPool4, 128, 256, Training, 'ConvLayer5')

		ConvLayer6 = ConvLayer(ConvLayer5, 256, 256, Training, 'ConvLayer6')

		ConvLayer7 = ConvLayer(ConvLayer6, 256, 256, Training, 'ConvLayer7')

		MPool7 = max_pool(ConvLayer7, ksize=(2, 1), stride=(2, 1))

		ConvLayer8 = ConvLayer(MPool7, 256, 512, Training, 'ConvLayer8')

		ConvLayer9 = ConvLayer(ConvLayer8, 512, 512, Training, 'ConvLayer9')

		ConvLayer10 = ConvLayer(ConvLayer9, 512, 512, Training, 'ConvLayer10')

		MPool10 = max_pool(ConvLayer10, ksize=(2, 1), stride=(2, 1))

		ConvLayer11 = ConvLayer(MPool10, 512, 512, Training, 'ConvLayer11')

		ConvLayer12 = ConvLayer(ConvLayer11, 512, 512, Training, 'ConvLayer12')

		ConvLayer13 = ConvLayer(ConvLayer12, 512, LastFilters, Training, 'ConvLayer13')

		MPool13 = max_pool(ConvLayer13, ksize=(2, 1), stride=(2, 1))

		MPool13_T = tf.transpose(MPool13, perm=[0,2,1,3])

		MPool13_T_RSH = tf.reshape(MPool13_T, [-1, FV, LastFilters])

		return tf.reshape(MPool13_T_RSH, [-1, NFeatures])

def ConvLayer(Input, FilterIn, FilterOut, Training, Scope):

	with tf.variable_scope(Scope):

		Weight = weight_variable([3, 3, FilterIn, FilterOut])

		if cfg.LeakyReLU == True:

			return tf.nn.leaky_relu(batch_norm_conv(conv2d(Input, Weight), FilterOut, Training))
		else:
			return tf.nn.relu(batch_norm_conv(conv2d(Input, Weight), FilterOut, Training))

