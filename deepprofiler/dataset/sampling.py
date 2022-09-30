import pandas as pd
import numpy as np
import skimage.io
import threading
import tqdm
import os
import shutil

import tensorflow as tf

import deepprofiler.imaging.boxes
import deepprofiler.imaging.cropping


class SingleCellSampler(deepprofiler.imaging.cropping.CropGenerator):

    def start(self, session):
        self.all_metadata = []
        self.session = session
        # Define input data batches
        with tf.compat.v1.variable_scope("train_inputs"):
            self.config["train"]["model"]["params"]["batch_size"] = self.config["train"]["validation"]["batch_size"]
            if self.config["prepare"].get("outlines") is not None:
                self.build_input_graph(export_masks=True)
            else:
                self.build_input_graph(export_masks=False)

    def process_batch(self, batch):
        for i in range(len(batch["keys"])):
            batch["locations"][i]["Key"] = batch["keys"][i].replace('-', '/')
            batch["locations"][i]["Target"] = batch["targets"][i][0]
            batch["locations"][i][self.dset.targets[0].field_name] = self.dset.targets[0].values[batch["targets"][i][0]]
            batch["locations"][i][self.config["train"]["partition"]["split_field"]] = batch["split"][i]

        metadata = pd.concat(batch["locations"])
        cols = ["Key", "Target", "Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"]
        seps = ["/", "@", "x", ".png"]
        metadata["Image_Name"] = ""

        for c in range(len(cols)):
            metadata["Image_Name"] += metadata[cols[c]].astype(str) + seps[c]

        boxes, box_ind, targets, masks = deepprofiler.imaging.boxes.prepare_boxes(batch, self.config)

        feed_dict = {
            self.input_variables["image_ph"]: batch["images"],
            self.input_variables["boxes_ph"]: boxes,
            self.input_variables["box_ind_ph"]: box_ind,
            self.input_variables["mask_ind_ph"]: masks
        }
        for i in range(len(targets)):
            tname = "target_" + str(i)
            feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

        output = self.session.run(self.input_variables["labeled_crops"], feed_dict)
        return output[0], metadata.reset_index(drop=True)

    def export_single_cells(self, key, image_array, meta):
        outdir = self.config["paths"]["single_cell_set"]
        key = self.dset.keyGen(meta)
        batch = {"keys": [key], "images": [image_array], "targets": [], "locations": [], "split": []}
        batch["locations"].append(deepprofiler.imaging.boxes.get_locations(key, self.config))
        batch["targets"].append([t.get_values(meta) for t in self.dset.targets])
        batch["split"].append(meta[self.config["train"]["partition"]["split_field"]])
        crops, metadata = self.process_batch(batch)
        empty_crops_indices = []
        for j in range(crops.shape[0]):
            if np.isnan(crops[j, :, :, :]).any():
                empty_crops_indices.append(j)
                print('The image {} was empty, skipping'.format(metadata.loc[j, "Image_Name"]))
                continue
            plate, well, site, name = metadata.loc[j, "Image_Name"].split('/')
            os.makedirs(os.path.join(outdir, plate, well, site), exist_ok=True)
            image = deepprofiler.imaging.cropping.unfold_channels(crops[j, :, :, :])
            skimage.io.imsave(os.path.join(outdir, metadata.loc[j, "Image_Name"]), image)

        metadata.drop(empty_crops_indices, inplace=True)
        metadata.reset_index(inplace=True)
        self.all_metadata.append(metadata)
        print("{}: {} single cells".format(key, crops.shape[0]))


def start_session():
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    main_session = tf.compat.v1.Session(config=configuration)
    tf.compat.v1.keras.backend.set_session(main_session)
    return main_session


def is_directory_empty(outdir):
    # Verify that the output directory is empty
    os.makedirs(outdir, exist_ok=True)
    files = os.listdir(outdir)
    if len(files) > 0:
        erase = ""
        while erase != "y" and erase != "n":
            erase = input("Delete " + str(len(files)) + " existing files in " + outdir + "? (y/n) ")
            print(erase)
        if erase == "n":
            print("Terminating export.")
            return False
        elif erase == "y":
            print("Removing previous files")
            shutil.rmtree(outdir)
    return True


def export_dataset(config, dset):
    outdir = config["paths"]["single_cell_set"]
    if not is_directory_empty(outdir):
        return

    session = start_session()
    cropper = SingleCellSampler(config, dset)
    cropper.start(session)
    dset.scan(cropper.export_single_cells, frame="all")
    df = pd.concat(cropper.all_metadata).reset_index(drop=True)
    df.to_csv(os.path.join(outdir, "sc-metadata.csv"), index=False)
    print("Exporting: done")    


