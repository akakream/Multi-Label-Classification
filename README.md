# MultiLabel-UC-Merced

usage: dct.py [-h] -d DATA_DESTINATION -s SHAPE -c CLASSES [-b BATCH_SIZE]  [-e EPOCHS]

optional arguments:
-h, --help            show this help message and exit
-d DATA_DESTINATION, --data_destination DATA_DESTINATION relative destination to the folder where pickle files are located
-s SHAPE, --shape SHAPE Input shape for the model. It is usually (256,256,3), if it is rgb
-c CLASSES, --classes CLASSES Number of classes. This is going to be added to the last layer of the model
-b BATCH_SIZE, --batch_size BATCH_SIZE Batch size, default is 64
-e EPOCHS, --epochs EPOCHS Number of epochs, default is 10

# Example

`python resnet.py -d=./../data/pickleApperal -s=256,256,3 -c=11 -b=64 -e=6` 
