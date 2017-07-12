import tensorflow as tf

import dataset.utils
import learning.models
import learning.cropping


def tmpf(a,b,c):
    print (a,b.shape,c)


def validate(config, dset):
    
    dset.scan(tmpf, frame="val")
    #net = learning.models.create_resnet(inputs, num_classes)
    print("Validate: done")


