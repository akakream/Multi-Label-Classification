import os

'''
Put this file under codeUcMerced/ucMerced/UCMerced_LandUse
'''

trainingFolder = './Splits/training'
testFolder = './Splits/test'
validationFolder = './Splits/validation'

for folder in os.listdir(trainingFolder):
	for filename in os.listdir(f'{trainingFolder}/{folder}')[-30:]:
		print(filename)
		os.remove(f'{trainingFolder}/{folder}/{filename}')

for folder in os.listdir(testFolder):
	for filename in os.listdir(f'{testFolder}/{folder}')[:-30]:
		os.remove(f'{testFolder}/{folder}/{filename}')
	for filename in os.listdir(f'{testFolder}/{folder}')[-15:]:
		os.remove(f'{testFolder}/{folder}/{filename}')

for folder in os.listdir(validationFolder):
	for filename in os.listdir(f'{validationFolder}/{folder}')[:-15]:
		os.remove(f'{validationFolder}/{folder}/{filename}')
