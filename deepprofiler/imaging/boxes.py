import os

import numpy as np
import pandas as pd

X_KEY = "Nuclei_Location_Center_X"
Y_KEY = "Nuclei_Location_Center_Y"

#################################################
## BOUNDING BOX HANDLING
#################################################

def get_locations(image_key, config, random_sample=None, seed=None):
    if config["dataset"]["locations"]["mode"] == "single_cells":
        return get_single_cell_locations(image_key, config, random_sample, seed)
    elif config["dataset"]["locations"]["mode"] == "full_image":
        return get_full_image_locations(image_key, config, random_sample, seed)
    else:
        return None


def get_single_cell_locations(image_key, config, random_sample=None, seed=None):
    keys = image_key.split("/")
    locations_file = "{}/{}-{}.csv".format(
        keys[0],
        keys[1],
        "Nuclei"
    )
    locations_path = os.path.join(config["paths"]["locations"], locations_file)
    if os.path.exists(locations_path):
        locations = pd.read_csv(locations_path)
        if random_sample is not None and random_sample < len(locations):
            return locations.sample(random_sample, random_state=seed)
        else:
            return locations
    else:
        return pd.DataFrame(columns=[X_KEY, Y_KEY])


def get_full_image_locations(image_key, config, random_sample, seed):
    cols = config["dataset"]["images"]["width"]
    rows = config["dataset"]["images"]["height"]
    coverage = config["dataset"]["locations"]["area_coverage"]
    cols_margin = cols - int(cols * coverage)
    rows_margin = rows - int(rows * coverage)
 
    data = None
    if coverage == 1.0:
        data = [[rows/2, cols/2]]
    else:
        if random_sample is not None:
            cols_pos = np.random.randint(low=-cols_margin/2, high=cols_margin/2, size=random_sample) + cols/2
            rows_pos = np.random.randint(low=-rows_margin/2, high=rows_margin/2, size=random_sample) + rows/2
            data = [[cols_pos[i], rows_pos[i]] for i in range(random_sample)]
        elif random_sample is None:
            cols_pos = [cols/2 - cols_margin/2, cols/2 - cols_margin/2, cols/2 + cols_margin/2, cols/2 + cols_margin/2, cols/2]
            rows_pos = [rows/2 - rows_margin/2, rows/2 + rows_margin/2, rows/2 - rows_margin/2, rows/2 + rows_margin/2, rows/2]
            data = [[cols_pos[i], rows_pos[i]] for i in range(5)]

    return pd.DataFrame(data=data, columns=[X_KEY, Y_KEY])


def prepare_boxes(batch, config):
    if config["dataset"]["locations"]["mode"] == "single_cells":
        box_side = config["dataset"]["locations"]["box_size"]
        return prepare_cropping_regions(batch, config, box_side, box_side)
    elif config["dataset"]["locations"]["mode"] == "full_image":
        cols = config["dataset"]["images"]["width"]
        rows = config["dataset"]["images"]["height"]
        coverage = config["dataset"]["locations"]["area_coverage"]
        return prepare_cropping_regions(batch, config, int(cols * coverage), int(rows * coverage))
    else:
        return None


def prepare_cropping_regions(batch, config, box_width, box_height):
    locations_batch = batch["locations"]
    image_targets = batch["targets"]
    images = batch["images"]
    all_boxes = []
    all_indices = []
    all_targets = [[] for i in range(len(image_targets[0]))]
    all_masks = []
    index = 0

    for locations in locations_batch:
        # Collect and normalize boxes between 0 and 1
        boxes = np.zeros((len(locations), 4), np.float32)
        boxes[:,0] = locations[Y_KEY] - box_height/2
        boxes[:,1] = locations[X_KEY] - box_width/2
        boxes[:,2] = locations[Y_KEY] + box_height/2
        boxes[:,3] = locations[X_KEY] + box_width/2
        boxes[:,[0,2]] /= config["dataset"]["images"]["height"]
        boxes[:,[1,3]] /= config["dataset"]["images"]["width"]
        # Create indicators for this set of boxes, belonging to the same image
        box_ind = index * np.ones((len(locations)), np.int32)
        # Propagate the same labels to all crops
        for i in range(len(image_targets[index])):
            all_targets[i].append(image_targets[index][i] * np.ones((len(locations)), np.int32))
        # Identify object mask for each crop
        masks = np.zeros(len(locations), np.int32)
        if config["dataset"]["locations"]["mask_objects"]:
            i = 0
            for lkey in locations.index:
                y = int(locations.loc[lkey, Y_KEY])
                x = int(locations.loc[lkey, X_KEY])
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
