
class MetadataColumnTarget(object):

    def __init__(self, field_name, values):
        self.field_name = field_name
        self.index = {}
        self.values = values
        self.values.sort()
        for i in range(len(self.values)):
            self.index[self.values[i]] = i
        print(self.index)

    def get_values(self, record):
        value = record[self.field_name]
        return self.index[value]

    @property
    def shape(self):
        return [None, len(self.index)]
