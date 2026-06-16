"""Classification target encoding.

:class:`MetadataColumnTarget` maps a categorical metadata column (e.g.
``"Class"``) to integer indices so that labels can be one-hot encoded and
passed to the model.  One target must be added to every
:class:`~deepprofiler.dataset.image_dataset.ImageDataset` before profiling,
because :class:`~deepprofiler.profiling.Profile` reads
``dset.targets[0].shape[1]`` to determine the number of output classes when
building the model.
"""


class MetadataColumnTarget(object):
    """Encode a metadata column as a sorted integer index.

    Unique values found in ``values`` are sorted and assigned consecutive
    integer indices starting from 0.  :meth:`get_values` looks up the index
    for a single record at inference time.

    Args:
        field_name: Name of the metadata column (e.g. ``"Class"``).
        values: Array-like of all unique values that appear in the column.
            The sort order of this array determines the class indices.
    """

    def __init__(self, field_name, values):
        self.field_name = field_name
        self.index = {}
        self.values = values
        self.values.sort()
        for i in range(len(self.values)):
            self.index[self.values[i]] = i
        print(self.index)

    def get_values(self, record):
        """Return the integer class index for one metadata record.

        Args:
            record: A dict-like or Pandas Series with a key matching
                ``self.field_name``.

        Returns:
            Integer index in ``[0, num_classes)``.
        """
        value = record[self.field_name]
        return self.index[value]

    @property
    def shape(self):
        """``[None, num_classes]`` — mirrors Keras layer output shape convention."""
        return [None, len(self.index)]
