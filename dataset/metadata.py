import pandas as pd

import dataset.utils
import sqlite3


def parseDelimiter(delimiter):
    if delimiter == "blanks":
        return '\s+'
    elif delimiter == "tabs":
        return '\t'
    else:
        return ','

def conditionalWellName(row):
    if row["Metadata_Plate"] in ["52650", "52661"]:
        return row["Metadata_Well"]
    else:
        return row["Metadata_Well"].upper()

## Generator of plates. Reads metadata and yields plates
def readPlates(metaFile):
    metadata = Metadata(metaFile)
    plates = metadata.data["Metadata_Plate"].unique()
    dataset.utils.logger.info("Total plates: {}".format(len(plates)))
    for i in range(len(plates)):  #  & (df.Metadata_Well == "a01")
        plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]) & (df.Metadata_Well == "a01"), copy=True)
        yield plate
    return

class Metadata():

    # The dtype argument indicates whether the dataset should be read as strings (object)
    # or according to the dataset type (None)
    def __init__(self, filename=None, csvMode="single", delimiter="default", dtype=object):
        if filename is not None:
            if csvMode == "single":
                self.loadSingle(filename, delimiter, dtype)
            elif csvMode == "multi":
                self.loadMultiple(filename, delimiter, dtype)
            print(self.data.info())

    def loadSingle(self, filename, delim, dtype):
        print("Reading metadata form", filename)
        delimiter = parseDelimiter(delim)
        # Read csv files as strings without dropping NA symbols
        self.data = pd.read_csv(filename, delimiter, dtype=dtype, keep_default_na=False)

    def loadMultiple(self, filename, delim, dtype):
        frames = []
        delimiter = parseDelimiter(delim)
        with open(filename, "r") as filelist:
            for line in filelist:
                csvPath = line.replace("\n","")
                print("Reading from", csvPath)
                frames.append( pd.read_csv(csvPath, delimiter, dtype=dtype, keep_default_na=False) )
        self.data = pd.concat(frames)
        print("Multiple CSV files loaded")

    def filterRecords(self, filteringRule, copy=False):
        if copy:
            newMeta = Metadata()
            newMeta.data = self.data.loc[filteringRule(self.data), :].copy()
            return newMeta
        else:
            self.data = self.data.loc[filteringRule(self.data), :]

    def splitMetadata(self, trainingRule, validationRule):
        self.train = self.data[trainingRule(self.data)].copy()
        self.val = self.data[validationRule(self.data)].copy()

def createCellFiles(args):
    plate, config = args

    plate_name = plate.data.iloc[0]["Metadata_Plate"]
    database_file = "{}/{}/{}.sqlite".format(config["original_images"]["backend"], plate_name, plate_name)

    conn = sqlite3.connect(database_file)
    query_template = "SELECT Cells.Cells_Location_Center_X,Cells_Location_Center_Y " +\
                     " FROM Cells LEFT JOIN Image " +\
                     "    ON Image.ImageNumber = Cells.ImageNumber " +\
                     " WHERE Image.Image_Metadata_Plate = '{}' " +\
                     "    AND Image.Image_Metadata_Well = '{}' " +\
                     "    AND Image.Image_Metadata_Site = '{}' "

    iteration = 1
    for index, row in plate.data.iterrows():
        # Read cells file for each image
        query = query_template.format(
            plate_name,
            row["Metadata_Well"],
            row["Metadata_Site"]
        )
        cell_locations = pd.read_sql_query(query, conn)

        # Keep center coordinates only, remove NaNs, and transform to integers
        cell_locations = cell_locations.dropna(axis=0, how="any")
        cell_locations *= config["compression"]["scaling_factor"]
        cell_locations["Cells_Location_Center_X"] = cell_locations["Cells_Location_Center_X"].astype(int)
        cell_locations["Cells_Location_Center_Y"] = cell_locations["Cells_Location_Center_Y"].astype(int)

        # Save the resulting dataset frame in the output directory
        cells_file = "{}/{}/cells/{}-{}.csv".format(
            config["compression"]["output_dir"],
            row["Metadata_Plate"],
            row["Metadata_Well"],
            row["Metadata_Site"]
        )
        dataset.utils.check_path(cells_file)
        cell_locations.to_csv(cells_file, index=False)
        dataset.utils.printProgress(iteration, len(plate.data), prefix='Cell locations in plate')
        iteration += 1

    print("")