"""Cell bounding box construction from centroid CSV files.

DeepProfiler does not perform cell segmentation.  Cell locations must be
provided externally (e.g. from CellProfiler) as CSV files stored under
``config["paths"]["locations"]`` with the naming convention::

    {plate}/{well}-{site}-Nuclei.csv

Each CSV must contain at minimum two columns:

- ``Nuclei_Location_Center_X`` — centroid X coordinate in pixels
- ``Nuclei_Location_Center_Y`` — centroid Y coordinate in pixels

:func:`get_locations` reads the appropriate CSV and returns a DataFrame of
centroids.  :func:`prepare_boxes` converts those centroids into normalised
``[y1, x1, y2, x2]`` bounding boxes suitable for
``tf.image.crop_and_resize``.
"""

import os

import numpy as np
import pandas as pd

X_KEY = "Nuclei_Location_Center_X"
Y_KEY = "Nuclei_Location_Center_Y"


def get_locations(image_key, config, random_sample=None, seed=None):
    """Return cell centroid locations for one image.

    Dispatches to :func:`get_single_cell_locations` (``mode: single_cells``)
    or :func:`get_full_image_locations` (``mode: full_image``) based on
    ``config["dataset"]["locations"]["mode"]``.

    Args:
        image_key: String key of the form ``"{plate}/{well}-{site}"`` that
            identifies the image within the dataset.
        config: Experiment configuration dict.
        random_sample: If not ``None``, randomly sample this many locations.
        seed: Random seed for reproducible sampling.

    Returns:
        DataFrame with at least ``Nuclei_Location_Center_X`` and
        ``Nuclei_Location_Center_Y`` columns, or an empty DataFrame if the
        locations file is missing.
    """
    if config["dataset"]["locations"]["mode"] == "single_cells":
        return get_single_cell_locations(image_key, config, random_sample, seed)
    elif config["dataset"]["locations"]["mode"] == "full_image":
        return get_full_image_locations(image_key, config, random_sample, seed)
    else:
        return None


def get_single_cell_locations(image_key, config, random_sample=None, seed=None):
    """Read per-cell centroids from the locations CSV for one image.

    Constructs the CSV path as::

        {config["paths"]["locations"]}/{plate}/{well}-{site}-Nuclei.csv

    Returns an empty DataFrame (with the expected columns) if the file does
    not exist, so the caller can safely check ``len(locations) == 0``.

    Args:
        image_key: String key ``"{plate}/{well}-{site}"``.
        config: Experiment configuration dict.
        random_sample: If not ``None`` and smaller than the number of cells,
            randomly sample this many rows.
        seed: Random seed for reproducible sampling.

    Returns:
        DataFrame of centroid coordinates.
    """
    keys = image_key.split("/")
    locations_file = "{}/{}-{}.csv".format(keys[0], keys[1], "Nuclei")
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
    """Generate a regular grid (or random sample) of crop centres for one image.

    Used when ``config["dataset"]["locations"]["mode"]`` is ``"full_image"``.
    If the view covers the whole image a single centre point is returned.
    Otherwise a grid of non-overlapping ``view_size × view_size`` tiles is
    produced (or a random sample of that many centres when ``random_sample``
    is set).

    Args:
        image_key: Ignored — grid is derived from config dimensions.
        config: Experiment configuration dict.  Uses ``dataset.images.width``,
            ``dataset.images.height``, and ``dataset.locations.view_size``.
        random_sample: Number of random centres to generate.  Pass ``None``
            for a deterministic grid.
        seed: Unused (grid generation is deterministic or uses numpy default
            RNG).

    Returns:
        DataFrame with ``Nuclei_Location_Center_X`` and
        ``Nuclei_Location_Center_Y`` columns.
    """
    cols = config["dataset"]["images"]["width"]
    rows = config["dataset"]["images"]["height"]
    view = config["dataset"]["locations"]["view_size"]
    assert (view <= cols) and (view <= rows)
    cols_margin = cols - view
    rows_margin = rows - view

    data = None
    if view == cols:
        data = [[cols / 2, rows / 2]]
    else:
        if random_sample is not None:
            cols_pos = np.random.randint(low=-cols_margin / 2, high=cols_margin / 2, size=random_sample) + cols / 2
            rows_pos = np.random.randint(low=-rows_margin / 2, high=rows_margin / 2, size=random_sample) + rows / 2
            data = [[cols_pos[i], rows_pos[i]] for i in range(random_sample)]
        elif random_sample is None:
            cols_pos = np.linspace(view / 2, cols - view / 2, int(np.ceil(cols / view)))
            rows_pos = np.linspace(view / 2, rows - view / 2, int(np.ceil(rows / view)))
            grid = np.meshgrid(rows_pos, cols_pos)
            rows_pos = grid[0].flatten()
            cols_pos = grid[1].flatten()
            data = [[rows_pos[i], cols_pos[i]] for i in range(len(cols_pos))]

    return pd.DataFrame(data=data, columns=[X_KEY, Y_KEY])


def prepare_boxes(batch, config):
    """Convert centroid locations to normalised bounding boxes for crop_and_resize.

    Dispatches to :func:`get_cropping_regions` with ``box_size`` set to
    ``dataset.locations.box_size`` (single-cell mode) or
    ``dataset.locations.view_size`` (full-image mode).

    Args:
        batch: Dict with keys ``"images"``, ``"locations"``, and ``"targets"``.
            ``locations`` is a list of DataFrames (one per image in the batch).
        config: Experiment configuration dict.

    Returns:
        Tuple ``(boxes, box_ind, targets, masks)`` ready to feed into the TF1
        crop graph placeholders.
    """
    if config["dataset"]["locations"]["mode"] == "single_cells":
        return get_cropping_regions(batch, config, config["dataset"]["locations"]["box_size"])
    elif config["dataset"]["locations"]["mode"] == "full_image":
        view = config["dataset"]["locations"]["view_size"]
        return get_cropping_regions(batch, config, view)
    else:
        return None


def get_cropping_regions(batch, config, box_size):
    """Build normalised bounding boxes from centroid coordinates.

    For each centroid ``(x, y)``, computes a square bounding box::

        [y - box_size/2,  x - box_size/2,  y + box_size/2,  x + box_size/2]

    Coordinates are normalised to ``[0, 1]`` by dividing Y by image height
    and X by image width, as required by ``tf.image.crop_and_resize``.  Crops
    that extend beyond the image boundary are automatically zero-padded by TF.

    Also reads the object mask label (the pixel value of the last channel at
    the cell centroid) when ``mask_objects`` is enabled in config.

    Args:
        batch: Dict with ``"images"``, ``"locations"``, and ``"targets"``.
        config: Experiment configuration dict.
        box_size: Side length of the bounding box in pixels.

    Returns:
        Tuple of four arrays:

        - ``boxes``: float32 array of shape ``(total_cells, 4)`` with
          normalised ``[y1, x1, y2, x2]`` coordinates.
        - ``box_ind``: int32 array of shape ``(total_cells,)`` mapping each
          box to its image index in the batch.
        - ``targets``: list of int32 arrays, one per target, each of shape
          ``(total_cells,)``.
        - ``masks``: int32 array of shape ``(total_cells,)`` with the object
          mask label for each cell (0 when masking is disabled).
    """
    locations_batch = batch["locations"]
    image_targets = batch["targets"]
    images = batch["images"]
    all_boxes = []
    all_indices = []
    all_targets = [[] for i in range(len(image_targets[0]))]
    all_masks = []
    index = 0

    for locations in locations_batch:
        boxes = np.zeros((len(locations), 4), np.float32)
        boxes[:, 0] = locations[Y_KEY] - box_size / 2
        boxes[:, 1] = locations[X_KEY] - box_size / 2
        boxes[:, 2] = locations[Y_KEY] + box_size / 2
        boxes[:, 3] = locations[X_KEY] + box_size / 2
        boxes[:, [0, 2]] /= config["dataset"]["images"]["height"]
        boxes[:, [1, 3]] /= config["dataset"]["images"]["width"]

        box_ind = index * np.ones((len(locations)), np.int32)

        for i in range(len(image_targets[index])):
            all_targets[i].append(image_targets[index][i] * np.ones((len(locations)), np.int32))

        masks = np.zeros(len(locations), np.int32)
        if config["dataset"]["locations"]["mask_objects"]:
            i = 0
            for lkey in locations.index:
                y = int(locations.loc[lkey, Y_KEY])
                x = int(locations.loc[lkey, X_KEY])
                patch = images[index][max(y - 5, 0):y + 5, max(x - 5, 0):x + 5, -1]
                if np.size(patch) > 0:
                    masks[i] = int(np.median(patch))
                i += 1

        all_boxes.append(boxes)
        all_indices.append(box_ind)
        all_masks.append(masks)
        index += 1

    result = (
        np.concatenate(all_boxes),
        np.concatenate(all_indices),
        [np.concatenate(t) for t in all_targets],
        np.concatenate(all_masks),
    )
    return result
