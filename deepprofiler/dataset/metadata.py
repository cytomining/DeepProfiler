"""Metadata loading and plate-level iteration.

DeepProfiler expects a metadata index CSV with at minimum these columns:

- ``Metadata_Plate`` — plate identifier
- ``Metadata_Well``  — well identifier
- ``Metadata_Site``  — site/field-of-view identifier
- One column per imaging channel listing the filename relative to the images root
- A label column (e.g. ``Class``) used as the classification target

:class:`Metadata` wraps a single such CSV (or a list of CSVs) and provides
filtering and train/val splitting.  :func:`read_plates` yields one
:class:`Metadata` object per plate, which is the interface used by the
``prepare`` pipeline.
"""

import pandas as pd

import deepprofiler.dataset.utils


def parse_delimiter(delimiter):
    """Translate a human-readable delimiter name to a pandas ``sep`` string.

    Args:
        delimiter: ``"blanks"`` for whitespace-separated, ``"tabs"`` for
            tab-separated, or any other value for comma-separated (default CSV).

    Returns:
        A string suitable for passing as ``sep`` to ``pd.read_csv``.
    """
    if delimiter == "blanks":
        return r"\s+"
    elif delimiter == "tabs":
        return "\t"
    else:
        return ","


def read_plates(metaFile):
    """Yield one :class:`Metadata` slice per plate from a metadata index CSV.

    Reads the full index, finds unique ``Metadata_Plate`` values, and yields
    a filtered :class:`Metadata` object for each plate in order.  Used by the
    ``prepare`` command to process illumination statistics and compression
    plate-by-plate.

    Args:
        metaFile: Path to the metadata index CSV.

    Yields:
        :class:`Metadata` containing only rows for one plate.
    """
    metadata = Metadata(metaFile)
    plates = metadata.data["Metadata_Plate"].unique()
    deepprofiler.dataset.utils.logger.info("Total plates: {}".format(len(plates)))
    for i in range(len(plates)):
        plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]), copy=True)
        yield plate


class Metadata():
    """Wrapper around a metadata index CSV (or list of CSVs).

    Stores the full table in ``self.data`` and provides helpers for
    filtering rows and splitting into train/val subsets.

    Args:
        filename: Path to a CSV file, or a text file listing CSV paths
            (one per line) when ``csvMode="multi"``.  Pass ``None`` to create
            an empty instance (used internally by :meth:`filterRecords`).
        csvMode: ``"single"`` (default) or ``"multi"``.
        delimiter: ``"blanks"``, ``"tabs"``, or anything else for commas.
            See :func:`parse_delimiter`.
        dtype: Passed to ``pd.read_csv``.  Use ``object`` (default) to read
            all columns as strings, or ``None`` to infer types.
    """

    def __init__(self, filename=None, csvMode="single", delimiter="default", dtype=object):
        if filename is not None:
            if csvMode == "single":
                self.loadSingle(filename, delimiter, dtype)
            elif csvMode == "multi":
                self.loadMultiple(filename, delimiter, dtype)

    def loadSingle(self, filename, delim, dtype):
        """Load a single CSV into ``self.data``."""
        print("Reading metadata form", filename)
        delimiter = parse_delimiter(delim)
        self.data = pd.read_csv(filename, sep=delimiter, dtype=dtype, keep_default_na=False)

    def loadMultiple(self, filename, delim, dtype):
        """Load a list of CSVs and concatenate them into ``self.data``.

        Args:
            filename: Path to a text file where each line is a CSV path.
            delim: Delimiter hint passed to :func:`parse_delimiter`.
            dtype: Type argument forwarded to ``pd.read_csv``.
        """
        frames = []
        delimiter = parse_delimiter(delim)
        with open(filename, "r") as filelist:
            for line in filelist:
                csvPath = line.replace("\n", "")
                print("Reading from", csvPath)
                frames.append(pd.read_csv(csvPath, sep=delimiter, dtype=dtype, keep_default_na=False))
        self.data = pd.concat(frames)
        print("Multiple CSV files loaded")

    def filterRecords(self, filteringRule, copy=False):
        """Keep only rows that satisfy ``filteringRule``.

        Args:
            filteringRule: Callable that takes the DataFrame and returns a
                boolean Series.
            copy: If True, return a new :class:`Metadata` with the filtered
                rows; if False, filter ``self.data`` in place.

        Returns:
            New :class:`Metadata` when ``copy=True``, otherwise ``None``.
        """
        if copy:
            newMeta = Metadata()
            newMeta.data = self.data.loc[filteringRule(self.data), :].copy()
            return newMeta
        else:
            self.data = self.data.loc[filteringRule(self.data), :]

    def splitMetadata(self, trainingRule, validationRule):
        """Populate ``self.train`` and ``self.val`` DataFrame subsets.

        Args:
            trainingRule: Callable ``(DataFrame) -> bool Series`` selecting
                training rows.
            validationRule: Callable ``(DataFrame) -> bool Series`` selecting
                validation rows.
        """
        self.train = self.data[trainingRule(self.data)].copy()
        self.val = self.data[validationRule(self.data)].copy()

    def mergeOutlines(self, outlines_df):
        """Join an outlines DataFrame onto the metadata on plate/well/site keys.

        Args:
            outlines_df: DataFrame with ``Metadata_Plate``, ``Metadata_Well``,
                and ``Metadata_Site`` columns plus an ``Outlines`` filename
                column.
        """
        result = pd.merge(self.data, outlines_df, on=["Metadata_Plate", "Metadata_Well", "Metadata_Site"])
        print("Metadata merged with Outlines")
        self.data = result
