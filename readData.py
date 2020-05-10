import os
import pandas as pd
# from PIL import Image
import cv2
import numpy as np
import random
import pickle

def readData():

	X_TRAIN = []
	Y_TRAIN = []

	X_TEST = []
	Y_TEST = []

	X_VALIDATION = []
	Y_VALIDATION = []

	training_data = []
	test_data = []
	validation_data = []

	# reads the excel file into df
	excel_file_path = './ucMerced/multilabels/LandUse_Multilabeled.xlsx'
	df = pd.read_excel(excel_file_path)

	# reads every row in the excel file into Rows array
	Rows = []
	for index, row in df.iterrows():
		row_list = [row['image'], row['airplane'], row['bare-soil'], row['buildings'], row['cars'], row['chaparral'],	row['court'],	row['dock'],	row['field'],	row['grass'],	row['mobile-home'],	row['pavement'],	row['sand'],	row['sea'],	row['ship'],	row['tanks'],	row['trees'],	row['water']]
		Rows.append(row_list)

	# iterates through every file under the given path and finds the paths of the filenames given in the excel file 
	for root, dirs, files in os.walk("./ucMerced/Splits"):
		for name in files:
			for row in Rows:
				if row[0] == name[:-4]: # if the file is found
					row.append(os.path.join(root, name)) # path of the image
					row.append(root.split('/')[-2]) # training, test or validation?

	# Prepared data can be written in an .xlsl file here actually

	for row in Rows:
		# im = Image.open(row[-2])
		# imarray = np.array(im)
		imarray = cv2.imread(row[-2], cv2.IMREAD_COLOR) # cv2 reads it as BGR; does it make a difference for CNN? I dont know...
		im = cv2.resize(imarray, (256,256))
		if row[-1] == 'training':
			training_data.append([im, row[1:18]])
		elif row[-1] == 'test':
			test_data.append([im, row[1:18]])
		else:
			validation_data.append([im, row[1:18]])

	random.shuffle(training_data)
	random.shuffle(test_data)
	random.shuffle(validation_data)
	# print(training_data[0])
	# print(len(training_data))

	for features, labels in training_data:
		X_TRAIN.append(features)
		Y_TRAIN.append(labels)

	# X_TRAIN = np.array(X_TRAIN).reshape(-1, 256, 256, 3)
	X_TRAIN = np.array(X_TRAIN).reshape(-1, 256, 256, 3)
	Y_TRAIN = np.array(Y_TRAIN)

	for features, labels in test_data:
		X_TEST.append(features)
		Y_TEST.append(labels)

	X_TEST = np.array(X_TEST)
	Y_TEST = np.array(Y_TEST)

	for features, labels in validation_data:
		X_VALIDATION.append(features)
		Y_VALIDATION.append(labels)

	X_VALIDATION = np.array(X_VALIDATION)
	Y_VALIDATION = np.array(Y_VALIDATION)	

	pickle_out = open('X_TRAIN.pickle', 'wb')
	pickle.dump(X_TRAIN, pickle_out)
	pickle_out.close()

	pickle_out = open('Y_TRAIN.pickle', 'wb')
	pickle.dump(Y_TRAIN, pickle_out)
	pickle_out.close()

	pickle_out = open('X_TEST.pickle', 'wb')
	pickle.dump(X_TEST, pickle_out)
	pickle_out.close()

	pickle_out = open('Y_TEST.pickle', 'wb')
	pickle.dump(Y_TEST, pickle_out)
	pickle_out.close()

	pickle_out = open('X_VALIDATION.pickle', 'wb')
	pickle.dump(X_VALIDATION, pickle_out)
	pickle_out.close()

	pickle_out = open('Y_VALIDATION.pickle', 'wb')
	pickle.dump(Y_VALIDATION, pickle_out)
	pickle_out.close()

	print(X_TRAIN.shape)
	print(len(X_TRAIN))

	return (X_TRAIN, Y_TRAIN, X_TEST, Y_TRAIN, X_VALIDATION, Y_VALIDATION)

readData()
