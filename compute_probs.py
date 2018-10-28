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

if (os.path.exists(cfg.Probs) == False): os.makedirs(cfg.Probs)

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

logits = tf.nn.softmax(logits, dim=-1, name=None)

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

		Logits = session.run([logits], feed_dict=feed)
		del feed

		_,sLen,_,_ = np.shape(Logits)

		for i in range(0, cfg.BatchSize):

			fileIndex = cfg.BatchSize * batch + i
			filename = "./"+cfg.Probs+"/" + os.path.basename(FilesList[fileIndex].strip()) + ".txt"

			file = codecs.open(filename, "a", "utf-8")

			for seqn in range(0, sLen):

				seq = Logits[0][seqn][i]

				file.write(str(seq[NClasses-1]))
				file.write(" ")

				for c in range(0, NClasses-1):
					val = seq[c]
					file.write(str(val))
					file.write(" ")

				file.write("\n")

			file.close

		start += cfg.BatchSize
		end += cfg.BatchSize
		batch += 1

except (KeyboardInterrupt, SystemExit, Exception) as e:
	print("[Error/Interruption] %s" % str(e))
	print("Clossing TF Session...")
	session.close()
	print("Terminating Program...")
	sys.exit(0)


