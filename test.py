from __future__ import print_function
###
# Copyright 2018 Edgard Chammas. All Rights Reserved.
# Licensed under the Creative Commons Attribution-NonCommercial International Public License, Version 4.0.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/legalcode
###

#!/usr/bin/python

import tensorflow as tf
import sys
import os
import cv2
import numpy as np
import codecs
import math

try:
	reload(sys)  # Python 2
	sys.setdefaultencoding('utf8')
except NameError:
	pass         # Python 3

from config import cfg
from util import LoadClasses
from util import LoadModel
from util import ReadData
from util import LoadList
from cnn import CNN
from cnn import WND_HEIGHT
from cnn import WND_WIDTH
from cnn import MPoolLayers_H
from rnn import RNN


if cfg.WriteDecodedToFile == True:
	DecodeLog = codecs.open("decoded.txt", "w", "utf-8")

Classes = LoadClasses(cfg.CHAR_LIST)

NClasses = len(Classes)

FilesList = LoadList(cfg.TEST_LIST)

WND_SHIFT = WND_WIDTH - 2

VEC_PER_WND = WND_WIDTH / math.pow(2, MPoolLayers_H)

phase_train = tf.Variable(True, name='phase_train')

x = tf.placeholder(tf.float32, shape=[None, WND_HEIGHT, WND_WIDTH])

SeqLens = tf.placeholder(shape=[cfg.BatchSize], dtype=tf.int32)

x_expanded = tf.expand_dims(x, 3)

Inputs = CNN(x_expanded, phase_train, 'CNN_1')

logits = RNN(Inputs, SeqLens, 'RNN_1')

# CTC Beam Search Decoder to decode pred string from the prob map
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, SeqLens)

#Reading test data...
InputListTest, SeqLensTest, _ = ReadData(cfg.TEST_LOCATION, cfg.TEST_LIST, cfg.TEST_NB, WND_HEIGHT, WND_WIDTH, WND_SHIFT, VEC_PER_WND, '')

print('Initializing...')

session = tf.Session()

session.run(tf.global_variables_initializer())

LoadModel(session, cfg.SaveDir+'/')

try:
	session.run(tf.assign(phase_train, False))

	randIxs = range(0, len(InputListTest))

	start, end = (0, cfg.BatchSize)

	batch = 0
	while end <= len(InputListTest):
		batchInputs = []
		batchSeqLengths = []
		for batchI, origI in enumerate(randIxs[start:end]):
			batchInputs.extend(InputListTest[origI])
			batchSeqLengths.append(SeqLensTest[origI])

		feed = {x: batchInputs, SeqLens: batchSeqLengths}
		del batchInputs, batchSeqLengths

		Decoded = session.run([decoded], feed_dict=feed)[0]
		del feed

		trans = session.run(tf.sparse_tensor_to_dense(Decoded[0]))

		for i in range(0, cfg.BatchSize):

			fileIndex = cfg.BatchSize * batch + i
			filename = FilesList[fileIndex].strip()
			decodedStr = " "
			
			for j in range(0, len(trans[i])):
				if trans[i][j] == 0:					
					if (j != (len(trans[i]) - 1)):
						if trans[i][j+1] == 0: break
						else: decodedStr = "%s%s" % (decodedStr, Classes[trans[i][j]])
					else:
						break
				else:	
					if trans[i][j] == (NClasses - 2):
						if (j != 0): decodedStr = "%s " % (decodedStr)
						else: continue
					else:
						decodedStr = "%s%s" % (decodedStr, Classes[trans[i][j]])

			decodedStr = decodedStr.replace("<SPACE>", " ")

			decodedStr = filename + decodedStr[:] + "\n"
			if cfg.WriteDecodedToFile == True: DecodeLog.write(decodedStr)
			else: print(decodedStr, end=' ')

		start += cfg.BatchSize
		end += cfg.BatchSize
		batch += 1

	DecodeLog.close()

except (KeyboardInterrupt, SystemExit, Exception) as e:
	print("[Error/Interruption] %s" % str(e))
	print("Clossing TF Session...")
	session.close()
	print("Terminating Program...")
	sys.exit(0)


