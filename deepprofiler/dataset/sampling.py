import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import threading
import pickle

import deepprofiler.imaging.boxes
import deepprofiler.imaging.cropping

class SingleCellSampler(deepprofiler.imaging.cropping.CropGenerator):

    def start(self, session):
        self.session = session
        # Define input data batches
        with tf.variable_scope("train_inputs"):
            self.config["train"]["model"]["params"]["batch_size"] = self.config["train"]["validation"]["batch_size"]
            self.build_input_graph()

    def process_batch(self, batch):
       #images = np.reshape(batch["images"], self.input_variables["shapes"]["batch"])
       boxes, box_ind, targets, masks = deepprofiler.imaging.boxes.prepare_boxes(batch, self.config)

       feed_dict = {
           self.input_variables["image_ph"]:batch["images"],
           self.input_variables["boxes_ph"]:boxes,
           self.input_variables["box_ind_ph"]:box_ind,
           self.input_variables["mask_ind_ph"]:masks
       }
       for i in range(len(targets)):
           tname = "target_" + str(i)
           feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

       output = self.session.run(self.input_variables["labeled_crops"], feed_dict)
       output = {"image_batch": output[0], "target_0": output[1]}
       return output


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
   cropper = SingleCellSampler(config, dset)
   cropper.start(session)

   pointer = dset.batch_pointer
   total_single_cells = 0
   while dset.batch_pointer >= pointer:
       pointer = dset.batch_pointer
       batch = dset.get_train_batch(lock)
       sc = np.sum([len(x) for x in batch['locations']])
       ni = len(batch["images"])
       total_single_cells += sc
       print(dset.batch_pointer, "batch_images", ni, "batch cells", sc, "total cells", total_single_cells)
       crops = cropper.process_batch(batch)
       pickle.dump(crops, open(str(pointer) + "_crops_file.pkl", "wb"))

   print(dset.batch_pointer, total_single_cells)
