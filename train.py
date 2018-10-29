###
# Copyright 2018 Edgard Chammas. All Rights Reserved.
# Licensed under the Creative Commons Attribution-NonCommercial International Public License, Version 4.0.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/legalcode
###

#!/usr/bin/python

import tensorflow as tf
import sys
import cv2
import numpy as np
import codecs
import math

from config import cfg
from util import LoadModel
from util import SaveModel
from util import ReadData
from util import target_list_to_sparse_tensor
from cnn import CNN
from cnn import CNNLight
from cnn import WND_HEIGHT
from cnn import WND_WIDTH
from cnn import WND_SHIFT
from cnn import MPoolLayers_H
from rnn import RNN

VEC_PER_WND = WND_WIDTH / math.pow(2, MPoolLayers_H)

nTimesNoProgress = 0

currTrainLoss = 1e6; currValLoss = 1e6

totalIter = cfg.TRAIN_NB / cfg.BatchSize

LogFile = codecs.open(cfg.LogFile, "a")

phase_train = tf.Variable(True, name='phase_train')

x = tf.placeholder(tf.float32, shape=[None, WND_HEIGHT, WND_WIDTH])

SeqLens = tf.placeholder(shape=[cfg.BatchSize], dtype=tf.int32)

x_expanded = tf.expand_dims(x, 3)

#Inputs = CNNLight(x_expanded, phase_train, 'CNN_1')
Inputs = CNN(x_expanded, phase_train, 'CNN_1')

logits = RNN(Inputs, SeqLens, 'RNN_1')

# Target params
indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])
values = tf.placeholder(dtype=tf.int32, shape=[None])
shape = tf.placeholder(dtype=tf.int64,shape=[2])

# Make targets
targets = tf.SparseTensor(indices, values, shape)

# Compute Loss
losses = tf.nn.ctc_loss(targets, logits, SeqLens)

loss = tf.reduce_mean(losses)

TrainLoss_s = tf.summary.scalar('TrainLoss', loss)

# CTC Beam Search Decoder to decode pred string from the prob map
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, SeqLens)

predicted = tf.to_int32(decoded[0])

error_rate = tf.reduce_sum(tf.edit_distance(predicted, targets, normalize=False)) / tf.to_float(tf.size(targets.values))    

TrainError_s = tf.summary.scalar('TrainError', error_rate)

tvars = tf.trainable_variables()

grad, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), cfg.MaxGradientNorm)

optimizer = tf.train.AdamOptimizer(learning_rate=cfg.LearningRate)

train_step = optimizer.apply_gradients(zip(grad, tvars))

#These values are used to draw performance graphs. Updated after each epoch.
OverallTrainingLoss = tf.Variable(0, name='OverallTrainingLoss', dtype=tf.float32)
OverallTrainingError = tf.Variable(0, name='OverallTrainingError', dtype=tf.float32)
OverallValidationLoss = tf.Variable(0, name='OverallValidationLoss', dtype=tf.float32)
OverallValidationError = tf.Variable(0, name='OverallValidationError', dtype=tf.float32)
OverallTrainingLoss_s = tf.summary.scalar('OverallTrainingLoss', OverallTrainingLoss)
OverallTrainingError_s = tf.summary.scalar('OverallTrainingError', OverallTrainingError)
OverallValidationLoss_s = tf.summary.scalar('OverallValidationLoss', OverallValidationLoss)
OverallValidationError_s = tf.summary.scalar('OverallValidationError', OverallValidationError)

#Reading training data...
inputList, seqLens, targetList = ReadData(cfg.TRAIN_LOCATION, cfg.TRAIN_LIST, cfg.TRAIN_NB, WND_HEIGHT, WND_WIDTH, WND_SHIFT, VEC_PER_WND, cfg.TRAIN_TRANS)

#Reading validation data...
if (cfg.VAL_NB > 0): inputListVal, seqLensVal, targetListVal = ReadData(cfg.VAL_LOCATION, cfg.VAL_LIST, cfg.VAL_NB, WND_HEIGHT, WND_WIDTH, WND_SHIFT, VEC_PER_WND, cfg.VAL_TRANS)

# Starting everything...
LogFile.write("Initializing...\n\n")

session = tf.Session()

session.run(tf.global_variables_initializer())

LocalTrainSummary = tf.summary.merge([TrainLoss_s, TrainError_s])

OverallSummary = tf.summary.merge([OverallTrainingLoss_s, OverallTrainingError_s, OverallValidationLoss_s, OverallValidationError_s])

SummaryWriter = tf.summary.FileWriter(cfg.LogDir, session.graph)

if cfg.StartingEpoch != 0: LoadModel(session, cfg.SaveDir+'/')

try:
	for epoch in range(cfg.StartingEpoch, cfg.NEpochs):
		
		LogFile.write("######################################################\n")
		LogFile.write("Training Data\n")

		TrainingLoss = []
		TrainingError = []

		if cfg.RandomBatches == True: randIxs = np.random.permutation(len(inputList))
		else: randIxs = range(0, len(inputList))

		start, end = (0, cfg.BatchSize)

		session.run(tf.assign(phase_train, True))

		batch = 0
		while end <= len(inputList):

			batchInputs = []
			batchTargetList = []
			batchSeqLengths = []

			for batchI, origI in enumerate(randIxs[start:end]):
				batchInputs.extend(inputList[origI])
				batchTargetList.append(targetList[origI])
				batchSeqLengths.append(seqLens[origI])

			batchTargetSparse = target_list_to_sparse_tensor(batchTargetList)
			batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse

			feed = {x: batchInputs, SeqLens: batchSeqLengths, indices: batchTargetIxs, values: batchTargetVals, shape: batchTargetShape}
			del batchInputs, batchTargetIxs, batchTargetVals, batchTargetShape, batchSeqLengths

			_, summary, Losses, Loss, Error = session.run([train_step, LocalTrainSummary, losses, loss, error_rate], feed_dict=feed)
			del feed

			SummaryWriter.add_summary(summary, epoch*totalIter + batch)
			
			numberOfInfElements = np.count_nonzero(np.isinf(Losses))
			if numberOfInfElements > 0:
				LogFile.write("WARNING: INF VALUE(S) FOUND!\n")
				LogFile.write("%s\n" % (batchTargetList[np.where(np.isinf(Losses)==True)[0][0]]))
				LogFile.write("Losses\n")
				Losses = filter(lambda v: ~np.isinf(v), Losses)
				Loss = np.mean(Losses)		

			TrainingLoss.append(Loss)
			TrainingError.append(Error)

			LogFile.write("Epoch %d, Batch: %d, Loss: %.6f, Error: %.6f, " % (epoch, batch, Loss, Error))

			if currTrainLoss < Loss: LogFile.write("Bad\n")
			else: LogFile.write("Good\n")

			start += cfg.BatchSize
			end += cfg.BatchSize
			batch += 1

		TrainingLoss = np.mean(TrainingLoss)
		TrainingError = np.mean(TrainingError)

		LogFile.write("Training loss: %.6f, Training error: %.6f\n" % (TrainingLoss, TrainingError) )

		if TrainingLoss < currTrainLoss:
			currTrainLoss = TrainingLoss
			LogFile.write("Training imporving.\n")
		else:
			LogFile.write("Training not imporving.\n")

		if (epoch + 1) % cfg.SaveEachNEpochs == 0:
			SaveModel(session, cfg.SaveDir+'/'+cfg.ModelName, epoch)

		if (cfg.VAL_NB > 0):

			LogFile.write("\nValidation Data\n");

			session.run(tf.assign(phase_train, False))

			ValidationError = []	
			ValidationLoss = []

			randIxs = range(0, len(inputListVal))
			start, end = (0, cfg.BatchSize)

			batch = 0
			while end <= len(inputListVal):

				batchInputs = []
				batchTargetList = []
				batchSeqLengths = []

				for batchI, origI in enumerate(randIxs[start:end]):
					batchInputs.extend(inputListVal[origI])
					batchTargetList.append(targetListVal[origI])
					batchSeqLengths.append(seqLensVal[origI])

				batchTargetSparse = target_list_to_sparse_tensor(batchTargetList)
				batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
			
				feed = {x: batchInputs, SeqLens: batchSeqLengths, indices: batchTargetIxs, values: batchTargetVals, shape: batchTargetShape}
				del batchInputs, batchTargetIxs, batchTargetVals, batchTargetShape, batchSeqLengths

				Loss, Error = session.run([loss, error_rate], feed_dict=feed)
				del feed

				ValidationError.append(Error)
				ValidationLoss.append(Loss)

				LogFile.write("Batch: %d, Loss: %.6f, Error: %.6f\n" % (batch, Loss, Error))

				start += cfg.BatchSize
				end += cfg.BatchSize
				batch += 1

			ValidationLoss = np.mean(ValidationLoss)
			ValidationError = np.mean(ValidationError)

			LogFile.write("Validation loss: %.6f, Validation error: %.6f\n" % (ValidationLoss, ValidationError))

			feed = {OverallTrainingLoss: TrainingLoss, OverallTrainingError: TrainingError, OverallValidationLoss: ValidationLoss, OverallValidationError: ValidationError}
	  		
			SummaryWriter.add_summary(session.run([OverallSummary], feed_dict = feed)[0], epoch)
			del feed

			if ValidationLoss < currValLoss:
				LogFile.write("Validation improving.\n")
				nTimesNoProgress = 0
				currValLoss = ValidationLoss
			else:
				LogFile.write("Validation not improving.\n")
				nTimesNoProgress = nTimesNoProgress + 1
				if nTimesNoProgress == cfg.TrainThreshold:
					session.close()
					LogFile.write("No progress on validation. Terminating program.\n")
					sys.exit(0)

			LogFile.write("######################################################\n\n")

except (KeyboardInterrupt, SystemExit, Exception) as e:
	print("[Error/Interruption] %s\n" % str(e))
	LogFile.write("[Error/Interruption] %s\n" % str(e))
	LogFile.write("Clossing TF Session...\n")
	session.close()
	LogFile.write("Terminating Program...\n")
	LogFile.close()
	sys.exit(0)

