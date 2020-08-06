import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import threading

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
       b = dset.get_train_batch(lock)
       sc = np.sum([len(x) for x in b['locations']])
       total_single_cells += sc
       print(dset.batch_pointer, sc, "cells", total_single_cells)
       pointer = dset.batch_pointer

   print(dset.batch_pointer, total_single_cells)
