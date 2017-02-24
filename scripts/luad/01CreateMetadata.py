################################################################################
## Script for preparing LUAD metadata.
## Creates a single file with image paths and treatment labels
## 02/23/2017. Broad Institute of MIT and Harvard
################################################################################

import argparse
import data.metadata as meta
import data.utils as utils
import pandas as pd

def relativePaths(df, target, path, filename, root):
    df[target] = df[path].str.replace(root, "") + df[filename]
    return df.drop([path, filename], axis=1)

def processMetadata(plate_maps, barcode_file, csv_list, root):
    # Load plate maps data and create labels
    plateMaps = meta.Metadata(plate_maps, "multi", "tabs")
    maps = plateMaps.data
    maps["Allele"] = maps["NCBIGeneID"].astype(str) + "@" + maps["x_mutation_status"]
    maps["Gene"] = 0
    alleles = maps["Allele"].unique()
    genes = maps["NCBIGeneID"].unique()
    print("Unique alleles:", len(alleles))
    for i in range(len(alleles)):
        maps.loc[lambda df: df.Allele == alleles[i], "Allele"] = i
        utils.printProgress (i+1,len(alleles), prefix="Alleles")
    print("Unique genes:", len(genes))
    for i in range(len(genes)):
        maps.loc[lambda df: df.broad_sample == genes[i], "Gene"] = i
        utils.printProgress (i+1,len(genes), prefix="Genes")

    # Load barcodes and csv files
    barcodes = meta.Metadata(barcode_file, "single")
    load_data = meta.Metadata(csv_list, "multi")

    # Merge two frames: csvs + barcodes to attach gene layout to each image 
    columns = list(load_data.data.columns.values)
    metadata = pd.merge(load_data.data.drop(columns[13:],axis=1), barcodes.data, 
                        left_on=["Metadata_Plate"], right_on=["Assay_Plate_Barcode"], how="inner")
    del load_data, barcodes

    # Concatenate paths and filenames and make them relative
    metadata = relativePaths(metadata, "RNA", "PathName_OrigRNA", "FileName_OrigRNA", root)
    metadata = relativePaths(metadata, "ER", "PathName_OrigER", "FileName_OrigER", root)
    metadata = relativePaths(metadata, "AGP", "PathName_OrigAGP", "FileName_OrigAGP", root)
    metadata = relativePaths(metadata, "Mito", "PathName_OrigMito", "FileName_OrigMito", root)
    metadata = relativePaths(metadata, "DNA", "PathName_OrigDNA", "FileName_OrigDNA", root)
    print(metadata.info())

    # Merge two frames: metadata + plateMaps to attach treatment info to each image
    metadata = pd.merge(metadata, maps, 
                        left_on=["Plate_Map_Name", "Metadata_Well"], 
                        right_on=["plate_map_name", "well_position"], how="left")
    metadata = metadata.drop(["plate_map_name","well_position","broad_sample","NCBIGeneID","pert_type",
                              "PublicID","Transcript","VirusPlateName","well_position","x_mutation_status",
                              "broad_sample","pert_name"], axis=1)
    metadata["plate_well"] = metadata["Metadata_Plate"].astype(str) + "::" + metadata["Metadata_Well"]

    # Find replicate labels
    metadata["Allele_Replicate"] = 0
    replicateDistribution = {}
    for i in range(len(alleles)):
        mask1 = metadata["Allele"] == i
        wells = metadata[mask1]["plate_well"].unique()
        utils.printProgress(i + 1, len(alleles), "Replicates")
        replicate = 1
        for j in range(len(wells)):
            mask2 = metadata["plate_well"] == wells[j]
            metadata.loc[mask1 & mask2, "Allele_Replicate"] = replicate
            replicate += 1
        try: replicateDistribution[replicate-1] += 1
        except: replicateDistribution[replicate-1] = 1
    metadata = metadata.drop(["plate_well"], axis=1)
    print(replicateDistribution)
    print(metadata.info())

    # Save resulting metadata
    metadata.to_csv("metadata.csv", index=False)
    dframe = pd.DataFrame({"ID":pd.Series(range(len(alleles))), "Allele":pd.Series(alleles)})
    dframe.to_csv("alleles.csv", index=False)
    dframe = pd.DataFrame({"ID":pd.Series(range(len(genes))), "Gene":pd.Series(genes)})
    dframe.to_csv("genes.csv", index=False)


if __name__ == "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("plate_maps", help="File containing the list of plate-map files")
    parser.add_argument("barcode_file", help="File with plate names")
    parser.add_argument("csv_list", help="File with the list of load_data.csv files")
    parser.add_argument("root", help="Absolute root directory of the images")
    args = parser.parse_args()

    processMetadata(args.plate_maps, args.barcode_file, args.csv_list, args.root)
