import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import threading
import pickle

import deepprofiler.imaging.boxes
import deepprofiler.imaging.cropping

def start_session():
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    main_session = tf.Session(config=configuration)
    keras.backend.set_session(main_session)
    return main_session

def sample_dataset(config, dset):
   session = start_session()
   dset.show_setup()
   lock = threading.Lock()

   pointer = dset.batch_pointer
   total_single_cells = 0
   while dset.batch_pointer >= pointer:
       pointer = dset.batch_pointer
       batch = dset.get_train_batch(lock)
       sc = np.sum([len(x) for x in batch['locations']])
       total_single_cells += sc
       print(dset.batch_pointer, sc, "cells", total_single_cells)
       #pickle.dump(batch, open("batch_file.pkl", "wb"))

       if len(batch["images"]) == 0: continue
       #images = np.reshape(batch["images"], self.input_variables["shapes"]["batch"])
       boxes, box_ind, targets, masks = deepprofiler.imaging.boxes.prepare_boxes(batch, config)

       ## The following code does not work because this function is not a subclass of CropGenerator
       ## However, we are in the right track, because we want to take advantage of the randomized 
       ## and balanced functionalities of ImageDataset. Perhaps the right thing to do is to create
       ## the subclass or to move this code in replacement of CropGenerator altogether.

       feed_dict = {
           self.input_variables["image_ph"]:batch["images"],
           self.input_variables["boxes_ph"]:boxes,
           self.input_variables["box_ind_ph"]:box_ind,
           self.input_variables["mask_ind_ph"]:masks
       }
       for i in range(len(targets)):
           tname = "target_" + str(i)
           feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

       output = sess.run(self.train_variables, feed_dict)

   print(dset.batch_pointer, total_single_cells)
