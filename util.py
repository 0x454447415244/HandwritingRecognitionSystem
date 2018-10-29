###
# Copyright 2018 Edgard Chammas. All Rights Reserved.
# Licensed under the Creative Commons Attribution-NonCommercial International Public License, Version 4.0.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/legalcode
###

#!/usr/bin/python

import tensorflow as tf
import numpy as np
import cv2
import math
import os
import codecs
from config import cfg

def LoadList(path):
    with open(path) as vlist:
        return vlist.readlines()

#Ref: https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
def batch_norm_conv(x, n_out, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W, stride=(1, 1), padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)

def max_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')

#Ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
def target_list_to_sparse_tensor(targetList):
    indices = []
    vals = []

    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))

def LoadClasses(path):
    data = {}
    with codecs.open(path, 'r', encoding='utf-8') as cF:
	    data = cF.read().split('\n')
    return data

def LoadList(path):
    with open(path) as vlist:
	return vlist.readlines()

def LoadModel(session, path):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(path)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        print('Checkpoint restored')
    else:
        print('No checkpoint found')
        exit()

def SaveModel(session, filename, epoch):
    saver = tf.train.Saver()
    saver.save(session, filename, global_step=epoch)

def ReadData(filesLocation, filesList, numberOfFiles, WND_HEIGHT, WND_WIDTH, WND_SHIFT, VEC_PER_WND, transDir=''):

	seqLens = []
	inputList = []
	targetList = []

	with open(filesList) as listHandler:

		imgNbr = 0
		imageFiles = listHandler.readlines()[0:numberOfFiles]

		for imageFile in imageFiles:

			if filesLocation != '': tfile = imageFile.strip('\n')
			else: tfile = os.path.basename(imageFile.strip('\n'))

			################################################################
			# Adding transcriptions

			if transDir != '':

			    targetFile = transDir + "/" + tfile + cfg.LabelFileType

			    with open(targetFile) as f:

				    data = f.readlines()

				    if len(data) == 0:
					    targetList.append([])
				    else:
					    for i in range(len(data)):
						    targetData = np.fromstring(data[i], dtype=np.uint16, sep=' ')
						    targetList.append(targetData)

			################################################################
			# Gathering the length of each sequence

			if filesLocation != '': imageFilePath = filesLocation + "/" + tfile + cfg.ImageFileType
			else: imageFilePath = imageFile.strip('\n') + cfg.ImageFileType

			print "Reading " + imageFilePath

			image = cv2.imread(imageFilePath, cv2.IMREAD_GRAYSCALE)

			h, w = np.shape(image)

			if(h > WND_HEIGHT): factor = WND_HEIGHT/float(h)
			else: factor = 1.0

			image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)

			h, w = np.shape(image)

			winId = 0
			wpd = 0
			while True:

				s = (winId * WND_SHIFT)
				e = s + WND_WIDTH

				if e > w:
					sl = (winId+1) * VEC_PER_WND

					if transDir != '':
					    #Fix for small sequences
					    if(len(targetList[imgNbr]) > sl):
						    diff = len(targetList[imgNbr]) - sl
						    wpd = int(math.ceil(float(diff) / VEC_PER_WND))
						    sl = sl + wpd * VEC_PER_WND

					seqLens.append(sl)

					break

				winId = winId + 1

			################################################################
			# Adding features

			featuresSet = []

			winId = 0
			while True:

				s = (winId * WND_SHIFT)
				e = s + WND_WIDTH

				if e > w:
					pad = np.ones((h, (e - w)), np.uint8)*255
					wnd = image[:h,s:w]
					wnd = np.append(wnd, pad, axis=1)

					if h < WND_HEIGHT:
						pad = np.ones(((WND_HEIGHT - h), WND_WIDTH), np.uint8)*255
						wnd = np.append(pad, wnd, axis=0)

					featuresSet.append(wnd)

					#Fix for small sequences
					pad = np.ones((WND_HEIGHT, WND_WIDTH), np.uint8)*255

					for i in range(wpd): featuresSet.append(pad)

					break

				wnd = image[:h,s:e]

				if h < WND_HEIGHT:
					pad = np.ones(((WND_HEIGHT - h), WND_WIDTH), np.uint8)*255
					wnd = np.append(pad, wnd, axis=0)

				featuresSet.append(wnd)
				winId = winId + 1

			################################################################
			inputList.append(featuresSet)

			imgNbr = imgNbr + 1
			################################################################

	if transDir != '':
	    assert len(inputList) == len(targetList)

	return inputList, seqLens, targetList

