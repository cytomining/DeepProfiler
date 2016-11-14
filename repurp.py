import data.metadata as meta
import data.utils as utils
import argparse
import pandas as pd

def relativePaths(df, target, path, filename, root):
    df[target] = df[path].str.replace(root, "") + df[filename]
    return df.drop([path, filename], axis=1)

def processMetadata():
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("plate_maps", help="File containing the list of plate-map files")
    parser.add_argument("barcode_file", help="File with plate names")
    parser.add_argument("csv_list", help="File with the list of load_data.csv files")
    parser.add_argument("root", help="Absolute root directory of the images")
    args = parser.parse_args()

    # Load plate maps data and create labels
    plateMaps = meta.Metadata(args.plate_maps, "multi", "blanks")
    maps = plateMaps.data
    maps["Treatment"] = maps["broad_sample"] + "@" + maps["mmoles_per_liter"].astype(str)
    maps["Compound"] = 0
    treatments = maps["Treatment"].unique()
    compounds = maps["broad_sample"].unique()
    print "Unique treatments:",len(treatments)
    for i in range(len(treatments)):
        maps.loc[lambda df: df.Treatment == treatments[i], "Treatment"] = i
        utils.printProgress (i+1,len(treatments), prefix="Treatments")
    print "Unique compounds:",len(compounds)
    for i in range(len(compounds)):
        maps.loc[lambda df: df.broad_sample == compounds[i], "Compound"] = i
        utils.printProgress (i+1,len(compounds), prefix="Compounds")

    # Load barcodes and csv files
    barcodes = meta.Metadata(args.barcode_file, "single")
    load_data = meta.Metadata(args.csv_list, "multi")

    # Merge two frames: csvs + barcodes to attach compound layout to each image 
    columns = list(load_data.data.columns.values)
    metadata = pd.merge(load_data.data.drop(columns[13:],axis=1), barcodes.data, 
                        left_on=["Metadata_Plate"], right_on=["Assay_Plate_Barcode"], how="inner")
    metadata = metadata.drop(["Batch_Number", "Batch_Date", "Assay_Plate_Barcode"],axis=1)
    del load_data, barcodes

    # Concatenate paths and filenames and make them relative
    metadata = relativePaths(metadata, "RNA", "PathName_OrigRNA", "FileName_OrigRNA", args.root)
    metadata = relativePaths(metadata, "ER", "PathName_OrigER", "FileName_OrigER", args.root)
    metadata = relativePaths(metadata, "AGP", "PathName_OrigAGP", "FileName_OrigAGP", args.root)
    metadata = relativePaths(metadata, "Mito", "PathName_OrigMito", "FileName_OrigMito", args.root)
    metadata = relativePaths(metadata, "DNA", "PathName_OrigDNA", "FileName_OrigDNA", args.root)
    print metadata.info()

    # Merge two frames: metadata + plateMaps to attach treatment info to each image
    metadata = pd.merge(metadata, maps, 
                        left_on=["Compound_Plate_Map_Name", "Metadata_Well"], 
                        right_on=["plate_map_name", "well_position"], how="left")
    metadata = metadata.drop(["plate_map_name","well_position","broad_sample","mg_per_ml","mmoles_per_liter","solvent"], axis=1)
    metadata["plate_well"] = metadata["Metadata_Plate"] + metadata["Metadata_Well"]

    # Find and label replicates
    metadata["Treatment_Replicate"] = 0
    replicateDistribution = {}
    for i in range(len(treatments)):
        mask1 = metadata["Treatment"] == i
        wells = metadata[mask1]["plate_well"].unique()
        utils.printProgress (i+1,len(treatments), "Replicates")
        replicate = 1
        for j in range(len(wells)):
            mask2 = metadata["plate_well"] == wells[j]
            metadata.loc[mask1 & mask2, "Treatment_Replicate"] = replicate
            replicate += 1
        try: replicateDistribution[replicate] += 1
        except: replicateDistribution[replicate] = 1
    metadata = metadata.drop(["plate_well"], axis=1)
    print replicateDistribution
    print metadata.info()

    # Save resulting metadata
    metadata.to_csv("metadata.csv", index=False)
    dframe = pd.DataFrame({"ID":pd.Series(range(len(treatments))), "Treatment":pd.Series(treatments)})
    dframe.to_csv("treatments.csv", index=False)
    dframe = pd.DataFrame({"ID":pd.Series(range(len(compounds))), "Compound":pd.Series(compounds)})
    dframe.to_csv("compounds.csv", index=False)


if __name__ == "__main__":
    processMetadata()
