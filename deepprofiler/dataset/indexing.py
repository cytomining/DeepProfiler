"""Metadata index splitting for parallel profiling runs.

Large experiments may have thousands of images that are impractical to profile
in a single job.  :func:`split_index` partitions the metadata index by
plate/well into N roughly equal parts and writes them as numbered CSV files
(``index-000.csv``, ``index-001.csv``, …) alongside the original ``index.csv``.

Each part can then be profiled independently::

    deepprofiler --root=... --config=... profile --part=0
    deepprofiler --root=... --config=... profile --part=1
    ...

The ``split`` CLI command calls this function.
"""

import pandas as pd


def split_index(config, parts):
    """Partition the metadata index into ``parts`` equal-sized CSV files.

    Groups the index by ``(Metadata_Plate, Metadata_Well)`` and distributes
    those groups as evenly as possible across ``parts`` files.  Each output
    file contains the full rows (all columns) for its subset of plate/well
    combinations.

    Output files are written to the same directory as ``index.csv``
    (``config["paths"]["metadata"]``), named ``index-000.csv`` through
    ``index-{N-1:03d}.csv``.

    Args:
        config: Experiment configuration dict.  Uses
            ``config["paths"]["metadata"]`` to locate and write index files.
        parts: Number of parts to split into.
    """
    index = pd.read_csv(config["paths"]["metadata"] + "/index.csv")
    plate_wells = index.groupby(["Metadata_Plate", "Metadata_Well"]).count()["Metadata_Site"]
    plate_wells = plate_wells.reset_index().drop(["Metadata_Site"], axis=1)
    part_size = int(len(plate_wells) / parts)
    for i in range(parts):
        if i < parts - 1:
            df = plate_wells[i * part_size:(i + 1) * part_size]
        else:
            df = plate_wells[i * part_size:]
        df = pd.merge(index, df, on=["Metadata_Plate", "Metadata_Well"])
        df.to_csv(config["paths"]["metadata"] + "/index-{0:03d}.csv".format(i), index=False)
    print("All set")
