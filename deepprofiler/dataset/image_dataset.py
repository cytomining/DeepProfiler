"""Dataset abstraction over a metadata index and per-channel image files.

:class:`ImageDataset` is the central data container used throughout
DeepProfiler.  It wraps a :class:`~deepprofiler.dataset.metadata.Metadata`
object and provides two key interfaces:

- :meth:`ImageDataset.get_image_paths` — resolve per-channel file paths for
  one metadata record.
- :meth:`ImageDataset.scan` — iterate over images and call a processing
  function (used by :func:`~deepprofiler.profiling.profile` to drive feature
  extraction).

:func:`read_dataset` is the standard factory that reads a config dict and
returns a fully initialised :class:`ImageDataset`.

:class:`ImageLocations` is a helper used internally to load all cell centroid
CSVs in parallel before training — it is not used during profiling.
"""

import os

import numpy as np
import pandas as pd

import deepprofiler.dataset.metadata
import deepprofiler.dataset.pixels
import deepprofiler.dataset.target
import deepprofiler.dataset.utils
import deepprofiler.imaging.boxes


class ImageLocations(object):
    """Pre-load cell locations for a set of images in parallel.

    Collects image keys, paths, and target labels from a metadata DataFrame,
    then uses :class:`~deepprofiler.dataset.utils.Parallel` to read all
    location CSVs concurrently.  Used by
    :meth:`ImageDataset.prepare_training_locations`.

    Args:
        metadata_training: DataFrame slice (e.g. ``Metadata.train``) to index.
        getImagePaths: Callable ``(row) -> (key, paths, outlines)``.
        targets: List of
            :class:`~deepprofiler.dataset.target.MetadataColumnTarget` objects.
    """

    def __init__(self, metadata_training, getImagePaths, targets):
        self.keys = []
        self.images = []
        self.targets = []
        self.outlines = []
        for i, r in metadata_training.iterrows():
            key, image, outl = getImagePaths(r)
            self.keys.append(key)
            self.images.append(image)
            self.targets.append([t.get_values(r) for t in targets])
            self.outlines.append(outl)
        print("Reading single-cell locations")

    def load_loc(self, params):
        """Load the locations CSV for one image (worker function for Parallel).

        Args:
            params: ``[index, config]`` as passed by
                :class:`~deepprofiler.dataset.utils.Parallel`.

        Returns:
            DataFrame with centroid coordinates plus ``ID``, ``ImageKey``,
            ``ImagePaths``, ``Target``, and ``Outlines`` columns appended.
        """
        i, config = params
        loc = deepprofiler.imaging.boxes.get_locations(self.keys[i], config)
        loc["ID"] = loc.index
        loc["ImageKey"] = self.keys[i]
        loc["ImagePaths"] = "#".join(self.images[i])
        loc["Target"] = self.targets[i][0]
        loc["Outlines"] = self.outlines[i]
        print("Image", i, ":", len(loc), "cells", end="\r")
        return loc

    def load_locations(self, config):
        """Load all location CSVs in parallel and return a list of DataFrames.

        Args:
            config: Experiment configuration dict (used for worker count and
                passed through to :func:`~deepprofiler.imaging.boxes.get_locations`).

        Returns:
            List of DataFrames, one per image.
        """
        process = deepprofiler.dataset.utils.Parallel(config, numProcs=config["train"]["sampling"]["workers"])
        data = process.compute(self.load_loc, [x for x in range(len(self.keys))])
        process.close()
        return data


class ImageDataset():
    """Container for a metadata index and its associated image files.

    Provides path resolution, per-image scanning, and (for the training path)
    location pre-loading and batch sampling.  During profiling only
    :meth:`get_image_paths`, :meth:`scan`, :meth:`add_target`, and
    :meth:`number_of_records` are used.

    Args:
        metadata: :class:`~deepprofiler.dataset.metadata.Metadata` object.
        sampling_field: Metadata column used as the classification label
            (e.g. ``"Class"``).
        channels: List of metadata column names, one per imaging channel,
            whose values are filenames relative to ``dataRoot``.
        dataRoot: Root directory containing image files.
        keyGen: Callable ``(row) -> str`` that produces the image key used to
            look up location CSVs (typically
            ``"{Metadata_Plate}/{Metadata_Well}-{Metadata_Site}"``).
        config: Full experiment configuration dict.
    """

    def __init__(self, metadata, sampling_field, channels, dataRoot, keyGen, config):
        self.meta = metadata
        self.channels = channels
        self.root = dataRoot
        self.keyGen = keyGen
        self.sampling_field = sampling_field
        self.sampling_values = metadata.data[sampling_field].unique()
        self.targets = []
        self.outlines = None
        self.config = config

    def get_image_paths(self, r):
        """Resolve per-channel file paths and the image key for one metadata row.

        If a channel value is already an absolute directory path it is used
        as-is; otherwise the filename is joined to ``self.root``.

        Args:
            r: A row from ``self.meta.data`` (Pandas Series or dict-like).

        Returns:
            Tuple ``(key, image_paths, outlines)`` where ``key`` is the string
            identifier (e.g. ``"Plate1/A01-1"``), ``image_paths`` is a list
            of resolved file paths (one per channel), and ``outlines`` is
            either ``None`` or the path to the outline image for this site.
        """
        key = self.keyGen(r)
        list_images = [r[ch] for ch in self.channels]
        paths = [(os.path.split(r[ch]))[0] for ch in self.channels]
        image = [list_images[ch] if os.path.isdir(paths[ch]) else self.root + "/" + list_images[ch] for ch in range(len(paths))]
        outlines = self.outlines
        if outlines is not None:
            outlines = self.outlines + r["Outlines"]
        return (key, image, outlines)

    def prepare_training_locations(self):
        # Load single cell locations in one data frame
        image_loc = ImageLocations(self.meta.train, self.get_image_paths, self.targets)
        locations = image_loc.load_locations(self.config)
        locations = pd.concat(locations)

        # Group by image and count the number of single cells per image in the column ID
        self.training_images = locations.groupby(["ImageKey", "Target"])["ID"].count().reset_index()

        workers = self.config["train"]["sampling"]["workers"]
        batch_size = self.config["train"]["model"]["params"]["batch_size"]
        cache_size = self.config["train"]["sampling"]["cache_size"]
        self.sampling_factor = self.config["train"]["sampling"]["factor"]

        # Count the total number of single cells
        self.total_single_cells = len(locations)
        # Median number of images per class
        self.sample_images = int(np.median(self.training_images.groupby("Target").count()["ID"]))
        # Number of classes
        targets = len(self.training_images["Target"].unique())
        self.config["num_classes"] = targets
        # Median number of single cells per image (column ID has counts as a result of groupby above)
        self.sample_locations = int(np.median(self.training_images["ID"]))
        # Set the target of single cells per epoch asuming a balanced set
        self.cells_per_epoch = int(targets * self.sample_images * self.sample_locations * self.sampling_factor)
        # Number of images that each worker should load at a time
        self.images_per_worker = int(batch_size / workers)
        # Percent of all cells that will be loaded in memory at a given moment in the queue
        self.cache_coverage = 100*(cache_size / self.cells_per_epoch)
        # Number of gradient updates required to approximately use all cells in an epoch
        self.steps_per_epoch = int(self.cells_per_epoch / batch_size)

        self.data_rotation = 0
        self.cache_records = 0
        self.shuffle_training_images()


    def show_setup(self):
        print(" || => Total single cells:", self.total_single_cells)
        print(" || => Median # of images per class:", self.sample_images)
        print(" || => Number of classes:", len(self.training_images["Target"].unique()))
        print(" || => Median # of cells per image:", self.sample_locations)
        print(" || => Approx. cells per epoch (with balanced sampling):", self.cells_per_epoch)
        print(" || => Images sampled per worker:", self.images_per_worker)
        print(" || => Cache data coverage: {}%".format(int(self.cache_coverage)))
        print(" || => Steps per epoch:", self.steps_per_epoch)
 

    def show_stats(self): ## Deprecated?
        # Proportion of images loaded by workers from all images that they should load in one epoch (recall)
        worker_efficiency = int(100 * (self.data_rotation / self.training_sample.shape[0]))
        # Proportion of single cells placed in the cache from all those that should be used in one epoch
        cache_usage = int(100 * self.cache_records / self.cells_per_epoch)
        #print("Training set coverage: {}% (worker efficiency). Data rotation: {}% (cache usage).".format(
        #          worker_efficiency,
        #          cache_usage)
        #)
        self.data_rotation = 0
        self.cache_records = 0
        return {'worker_efficiency': worker_efficiency, 'cache_usage': cache_usage}

    def shuffle_training_images(self):
        # Images in the original metadata file are resampled at each epoch
        sample = []
        for c in self.meta.train[self.sampling_field].unique():
            # Sample the same number of images per class. Oversample if the class has less images than needed
            mask = self.meta.train[self.sampling_field] == c
            available = self.meta.train[mask].shape[0]
            rec = self.meta.train[mask].sample(n=self.sample_images, replace=available < self.sample_images)
            sample.append(rec)

        # Shuffle and restart pointers. Note that training sample has images instead of single cells.
        self.training_sample = pd.concat(sample)
        self.training_sample = self.training_sample.sample(frac=1.0).reset_index(drop=True)
        self.batch_pointer = 0

    def get_train_batch(self, lock):
        # Select the next group of available images for cropping
        lock.acquire()
        df = self.training_sample[self.batch_pointer:self.batch_pointer + self.images_per_worker].copy()
        self.batch_pointer += self.images_per_worker
        self.data_rotation += self.images_per_worker
        if self.batch_pointer > self.training_sample.shape[0]:
            self.shuffle_training_images()
        lock.release()

        # Prepare the batch and cropping information for these images
        batch = {"keys": [], "images": [], "targets": [], "locations": []}
        sample = max(1, int(self.sample_locations * self.sampling_factor))
        for k, r in df.iterrows():
            key, image, outl = self.get_image_paths(r)
            batch["keys"].append(key)
            batch["targets"].append([t.get_values(r) for t in self.targets])
            batch["images"].append(deepprofiler.dataset.pixels.openImage(image, outl))
            batch["locations"].append(deepprofiler.imaging.boxes.get_locations(key, self.config, random_sample=sample))

        return batch

    def scan(self, f, frame="train", check=lambda k: True):
        """Iterate over images and call ``f`` for each one that passes ``check``.

        This is the primary driver for both profiling and compression.  Images
        are loaded sequentially (not in parallel) using
        :func:`~deepprofiler.dataset.pixels.openImage`.

        Args:
            f: Callable ``(index, image_array, meta_row)`` invoked for each
                image.  ``image_array`` is a ``(H, W, C)`` numpy array.
            frame: Which subset to iterate: ``"all"`` for the full metadata,
                ``"val"`` for the validation split, or ``"train"`` (default)
                for the training split.
            check: Optional predicate ``(meta_row) -> bool``.  Images for
                which ``check`` returns ``False`` are skipped.  Defaults to
                always returning ``True``.  Used by
                :meth:`~deepprofiler.profiling.Profile.check` to skip
                already-profiled images.
        """
        if frame == "all":
            frame = self.meta.data.iterrows()
        elif frame == "val":
            frame = self.meta.val.iterrows()
        else:
            frame = self.meta.train.iterrows()

        images = [(i, self.get_image_paths(r), r) for i, r in frame]
        for img in images:
            index = img[0]
            meta = img[2]
            if check(meta):
                image = deepprofiler.dataset.pixels.openImage(img[1][1], img[1][2])
                f(index, image, meta)
        return

    def number_of_records(self, dataset):
        """Return the number of rows in the requested split.

        Args:
            dataset: ``"all"``, ``"train"``, or ``"val"``.

        Returns:
            Integer row count, or 0 for an unrecognised ``dataset`` value.
        """
        if dataset == "all":
            return len(self.meta.data)
        elif dataset == "val":
            return len(self.meta.val)
        elif dataset == "train":
            return len(self.meta.train)
        else:
            return 0

    def add_target(self, new_target):
        """Append a :class:`~deepprofiler.dataset.target.MetadataColumnTarget` to ``self.targets``."""
        self.targets.append(new_target)


def read_dataset(config, mode='train'):
    """Build an :class:`ImageDataset` from a config dict.

    Reads the metadata index CSV, optionally replaces ``.tif``/``.tiff``
    extensions with ``.png`` if image compression was applied, merges outline
    CSVs if specified, adds classification targets, and (for the training path)
    pre-loads all cell locations.

    Args:
        config: Experiment configuration dict.  Must contain at minimum
            ``paths.index``, ``dataset``, ``train.partition``, and
            ``prepare.compression`` sections.
        mode: ``"train"`` to split metadata and pre-load locations, or any
            other value (e.g. ``"profile"``) to skip those steps.

    Returns:
        Fully initialised :class:`ImageDataset`.
    """
    metadata = deepprofiler.dataset.metadata.Metadata(config["paths"]["index"], dtype=None)
    if config["prepare"]["compression"]["implement"]:
        metadata.data.replace({'.tiff': '.png', '.tif': '.png'}, inplace=True, regex=True)

    # Add outlines if specified
    outlines = None
    if "outlines" in config["prepare"].keys() and config["prepare"]["outlines"] != "":
        df = pd.read_csv(config["paths"]["metadata"] + "/outlines.csv")
        metadata.mergeOutlines(df)
        outlines = config["paths"]["root"] + "inputs/outlines/"

    print(metadata.data.info())

    # Split training data
    if mode == 'train' and config["train"]["model"]["crop_generator"] == 'crop_generator':
        split_field = config["train"]["partition"]["split_field"]
        trainingFilter = lambda df: df[split_field].isin(config["train"]["partition"]["training"])
        validationFilter = lambda df: df[split_field].isin(config["train"]["partition"]["validation"])
        metadata.splitMetadata(trainingFilter, validationFilter)


    # Create a dataset
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = ImageDataset(
        metadata,
        config["dataset"]["metadata"]["label_field"],
        config["dataset"]["images"]["channels"],
        config["paths"]["images"],
        keyGen,
        config
    )

    # Add training targets
    for t in config["train"]["partition"]["targets"]:
        new_target = deepprofiler.dataset.target.MetadataColumnTarget(t, metadata.data[t].unique())
        dset.add_target(new_target)

    # Activate outlines for masking if needed
    if config["dataset"]["locations"]["mask_objects"]:
        dset.outlines = outlines

    # For training with sampled_crop_generator, no need to read locations again.
    if mode == 'train' and config["train"]["model"]["crop_generator"] == 'crop_generator':
        dset.prepare_training_locations()

    return dset


