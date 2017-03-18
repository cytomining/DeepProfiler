import pandas as pd

import dataset.utils
import sqlite3


def parse_delimiter(delimiter):
    if delimiter == "blanks":
        return '\s+'
    elif delimiter == "tabs":
        return '\t'
    else:
        return ','

#TODO: This function is only useful for the LUAD dataset
def conditionalWellName(row):
    if row["Metadata_Plate"] in ["52650", "52661"]:
        return row["Metadata_Well"]
    else:
        return row["Metadata_Well"].upper()

## Generator of plates. Reads metadata and yields plates
def read_plates(metaFile):
    metadata = Metadata(metaFile)
    plates = metadata.data["Metadata_Plate"].unique()
    dataset.utils.logger.info("Total plates: {}".format(len(plates)))
    for i in range(len(plates)):  #  & (df.Metadata_Well == "a01")
        plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]), copy=True)
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
        delimiter = parse_delimiter(delim)
        # Read csv files as strings without dropping NA symbols
        self.data = pd.read_csv(filename, delimiter, dtype=dtype, keep_default_na=False)

    def loadMultiple(self, filename, delim, dtype):
        frames = []
        delimiter = parse_delimiter(delim)
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


def write_locations(field, query_template, plate_name, row, conn, config):
    # Read cells file for each image
    query = query_template.replace("@@@",field).format(
            plate_name,
            row["Metadata_Well"],
            row["Metadata_Site"]
    )
    locations = pd.read_sql_query(query, conn)

    # Keep center coordinates only, remove NaNs, and transform to integers
    locations = locations.dropna(axis=0, how="any")
    locations[field+"_Location_Center_X"] = locations[field+"_Location_Center_X"]*config["compression"]["scaling_factor"]
    locations[field+"_Location_Center_Y"] = locations[field+"_Location_Center_Y"]*config["compression"]["scaling_factor"]
    locations[field+"_Location_Center_X"] = locations[field+"_Location_Center_X"].astype(int)
    locations[field+"_Location_Center_Y"] = locations[field+"_Location_Center_Y"].astype(int)

    # Save the resulting dataset frame in the output directory
    loc_file = "{}/{}/locations/{}-{}-{}.csv".format(
        config["compression"]["output_dir"],
        row["Metadata_Plate"],
        row["Metadata_Well"],
        row["Metadata_Site"],
        field
    )
    dataset.utils.check_path(loc_file)
    locations.to_csv(loc_file, index=False)


def create_cell_indices(args):
    plate, config = args
 
    # Open database
    plate_name = plate.data.iloc[0]["Metadata_Plate"]
    database_file = "{}/{}/{}.sqlite".format(config["original_images"]["backend"], plate_name, plate_name)
    conn = sqlite3.connect(database_file)

    # Define query template: @@@ is either Cells or Nuclei
    query_template = "SELECT @@@_Location_Center_X, @@@_Location_Center_Y " +\
                     " FROM @@@ INNER JOIN Image " +\
                     "    ON Image.ImageNumber = @@@.ImageNumber " +\
                     "    AND Image.TableNumber = @@@.TableNumber " +\
                     " WHERE Image.Image_Metadata_Plate = '{}' " +\
                     "    AND Image.Image_Metadata_Well = '{}' " +\
                     "    AND Image.Image_Metadata_Site = '{}' " +\
                     "    AND @@@_Location_Center_X NOT LIKE 'NaN' " +\
                     "    AND @@@_Location_Center_Y NOT LIKE 'NaN' "

    # Extract cells and nuclei locations for each image
    iteration = 1
    for index, row in plate.data.iterrows():
        write_locations("Cells", query_template, plate_name, row, conn, config)
        write_locations("Nuclei", query_template, plate_name, row, conn, config)
        dataset.utils.printProgress(iteration, len(plate.data), prefix='Locations in plate ' + str(plate_name))
        iteration += 1
    print("")

