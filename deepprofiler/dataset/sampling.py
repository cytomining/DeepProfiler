import numpy as np
import pandas as pd
import skimage.io
import threading
import pickle

import tensorflow as tf
import tensorflow.keras as keras

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
        for i in range(len(batch["keys"])):
            batch["locations"][i]["Key"] = batch["keys"][i]
            batch["locations"][i]["Target"] = batch["targets"][i][0]
        metadata = pd.concat(batch["locations"])
        cols = ["Key","Target","Nuclei_Location_Center_X","Nuclei_Location_Center_Y"]
        seps = ["+","@","x",".png"]
        metadata["Image_Name"] = ""
        for c in range(len(cols)):
            metadata["Image_Name"] += metadata[cols[c]].astype(str).str.replace("/","-") + seps[c]
        
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
        return output[0], metadata.reset_index()


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
    all_metadata = []
    while dset.batch_pointer >= pointer:
        pointer = dset.batch_pointer
        batch = dset.get_train_batch(lock)
        crops, metadata = cropper.process_batch(batch)
        for j in range(crops.shape[0]):
            image = deepprofiler.imaging.cropping.unfold_channels(crops[j,:,:,:])
            skimage.io.imsave(metadata.loc[j, "Image_Name"], image)
        all_metadata.append(metadata)

        total_single_cells += len(metadata)
        print("Images:", len(metadata.Key.unique()), "Sampled cells:", len(metadata), "Total", total_single_cells)
    print("Total single cells sampled:", total_single_cells)

