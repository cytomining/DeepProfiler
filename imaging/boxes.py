import numpy as np
import pandas as pd
import os

#################################################
## BOUNDING BOX HANDLING
#################################################

def getLocations(image_key, config, randomize=True):
    keys = image_key.split("/")
    locations_file = "{}/locations/{}-{}.csv".format(
        keys[0],
        keys[1],
        config["sampling"]["locations_field"]
    )
    locations_path = os.path.join(config["image_set"]["path"], locations_file)
    if os.path.exists(locations_path):
        locations = pd.read_csv(locations_path)
        random_sample = config["sampling"]["locations"]
        if randomize and random_sample is not None and random_sample < len(locations):
            return locations.sample(random_sample)
        else:
            return locations
    else:
        y_key = config["sampling"]["locations_field"] + "_Location_Center_Y"
        x_key = config["sampling"]["locations_field"] + "_Location_Center_X"
        return pd.DataFrame(columns=[x_key, y_key])


def prepareBoxes(batch, config):
    locationsBatch = batch["locations"]
    image_targets = batch["targets"]
    images = batch["images"]
    all_boxes = []
    all_indices = []
    all_targets = [[] for i in range(len(image_targets[0]))]
    all_masks = []
    index = 0
    y_key = config["sampling"]["locations_field"] + "_Location_Center_Y"
    x_key = config["sampling"]["locations_field"] + "_Location_Center_X"
    for locations in locationsBatch:
        # Collect and normalize boxes between 0 and 1
        boxes = np.zeros((len(locations), 4), np.float32)
        boxes[:,0] = locations[y_key] - config["sampling"]["box_size"]/2
        boxes[:,1] = locations[x_key] - config["sampling"]["box_size"]/2
        boxes[:,2] = locations[y_key] + config["sampling"]["box_size"]/2
        boxes[:,3] = locations[x_key] + config["sampling"]["box_size"]/2
        boxes[:,[0,2]] /= config["image_set"]["height"]
        boxes[:,[1,3]] /= config["image_set"]["width"]
        # Create indicators for this set of boxes, belonging to the same image
        box_ind = index * np.ones((len(locations)), np.int32)
        # Propage the same labels to all crops
        for i in range(len(image_targets[index])):
            all_targets[i].append(image_targets[index][i] * np.ones((len(locations)), np.int32))
        # Identify object mask for each crop
        masks = np.zeros(len(locations), np.int32)
        if config["image_set"]["mask_objects"]:
            i = 0
            for lkey in locations.index:
                y = int(locations.loc[lkey, y_key])
                x = int(locations.loc[lkey, x_key])
                patch = images[index][max(y-5,0):y+5, max(x-5,0):x+5, -1]
                if np.size(patch) > 0:
                    masks[i] = int(np.median(patch))
                i += 1
        # Pile up the resulting variables
        all_boxes.append(boxes)
        all_indices.append(box_ind)
        all_masks.append(masks)
        index += 1

    result = (np.concatenate(all_boxes),
              np.concatenate(all_indices),
              [np.concatenate(t) for t in all_targets],
              np.concatenate(all_masks)
             )
    return result


def loadBatch(dataset, config):
    batch = dataset.getTrainBatch(config["sampling"]["images"])
    batch["locations"] = [ getLocations(x, config) for x in batch["keys"] ]
    return batch

