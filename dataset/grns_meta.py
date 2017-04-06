import json
import sys
import os
import scipy.io
import pandas
import json

from tqdm import tqdm

import dataset.metadata
import dataset.utils
import dataset.indexing


# Let's read and process 3 million image paths like a boss
# Process time: 5 seconds
def read_index(filename):
    contents = open(filename, "r")
    data = [l.replace("\n","") for l in contents.readlines()]
    index = {}
    for i in tqdm(range(len(data))):
        d = data[i].split("/")
        try: index[ d[-2] ][ d[-1] ] = i
        except: index[ d[-2] ] = { d[-1]: i }
    return index


def main(config):
    # Load csv metadata files
    config["metadata"]["csv_list"] = os.path.join(config["metadata"]["path"], config["metadata"]["csv_list"])
    load_data = dataset.metadata.Metadata(config["metadata"]["csv_list"], "multi")
    metadata = load_data.data
    metadata.rename(columns={"FileName_Alexa568":"FileName_OrigAlexa568", "PathName_Alexa568":"PathName_OrigAlexa568"}, inplace=True)

    # Load labels matrix
    print("Reading annotations matrix")
    M = scipy.io.mmread(config["metadata"]["labels_matrix"])
    O = M.todok()

    print("Reading image-annotation keys")
    # Index with keys=plate_id, and second key=image_name
    matrix_row_ids = read_index(config["metadata"]["labels_matrix_ids"])
    
    print("Reading cluster identifiers")
    clusters = [l.replace("\n","") for l in open(config["metadata"]["cluster_ids"], "r")]
    print()

    for ch in config["original_images"]["channels"]:
        metadata["PathName_Orig" + ch] = metadata["PathName_Orig" + ch].str.cat(["/"]*len(metadata))
        metadata = dataset.indexing.relative_paths(
            metadata,
            ch,
            "PathName_Orig" + ch,
            "FileName_Orig" + ch,
            config["original_images"]["path"]
        )

    # Merge data
    metadata.drop(["Metadata_AbsPositionZ","Metadata_AbsTime", "Metadata_BinningX", "Metadata_BinningY", 
                   "Metadata_ChannelID", "Metadata_ChannelName", "Metadata_Col", "Metadata_ExposureTime",
                   "Metadata_FieldID", "Metadata_ImageResolutionX", "Metadata_ImageResolutionY", "Metadata_ImageSizeX",
                   "Metadata_MaxIntensity", "Metadata_ObjectiveMagnification", "Metadata_PositionX", "Metadata_PositionY",
                   "Metadata_PositionY", "Metadata_Row", "Metadata_ImageSizeY", "Metadata_PositionZ"], 
                   axis=1, inplace=True)
    print("Unique plates:", len(metadata["Metadata_Plate"].unique()))
    metadata[config["metadata"]["label_field"]] = ""
    metadata[config["metadata"]["control_field"]] = ""
    metadata.index = range(len(metadata))

    annotated = 0
    for k in tqdm(range(len(metadata))):
        images = matrix_row_ids[ metadata.iloc[k]["Metadata_Plate"] ]
        img_name = metadata.iloc[k]["Hoechst"].split("/")
        try: row = images[img_name[-1]]
        except: row = -1
        if row > 0: 
            annotated += 1
            metadata.set_value(k, config["metadata"]["control_field"], clusters[row])
            labels = {int(e[0][1]): int(e[1]) for e in O[row].items()}
            metadata.set_value(k, config["metadata"]["label_field"], json.dumps(labels))
       
    print("Annotated images:", annotated)
    metadata.to_csv("metadata.csv")
    

# Main data processing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: grns_meta.py config_file.json")
        sys.exit()

    config_file = open(sys.argv[1], "r")
    config = json.load(config_file)
    main(config)
