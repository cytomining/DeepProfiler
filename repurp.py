import data.metadata as meta
import argparse
import pandas as pd

def relativePaths(df, target, path, filename, root):
    df[target] = df[path].str.replace(root, '') + df[filename]
    return df.drop([path, filename], axis=1)

def run():
    # Parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("plate_maps", help="File containing the list of plate-map files")
    parser.add_argument("barcode_file", help="File with plate names")
    parser.add_argument("csv_list", help="File with the list of load_data.csv files")
    parser.add_argument("root", help="Absolute root directory of the images")
    args = parser.parse_args()
    # Create DataFrames using the Metadata class
    plateMaps = meta.Metadata(args.plate_maps, "multi", "blanks")
    maps = plateMaps.data
    maps['treatment'] = maps['broad_sample'] + "@" + maps['mmoles_per_liter'].round(4).astype(str)
    print maps.info()
    print len( maps['treatment'].unique() )
    print len( maps['broad_sample'].unique() )
    print len( maps['mmoles_per_liter'].round(4).unique() )
    return
    barcodes = meta.Metadata(args.barcode_file, "single")
    load_data = meta.Metadata(args.csv_list, "multi")
    # Merge two frames: csvs + barcodes to attach compound layout to each image 
    columns = list(load_data.data.columns.values)
    metadata = pd.merge(load_data.data.drop(columns[13:],axis=1), barcodes.data, 
                        left_on=["Metadata_Plate"], right_on=["Assay_Plate_Barcode"], how="inner")
    metadata = metadata.drop(['Batch_Number', 'Batch_Date', 'Assay_Plate_Barcode'],axis=1)
    del load_data, barcodes
    # Concatenate paths and filenames and make them relative
    metadata = relativePaths(metadata, 'RNA', 'PathName_OrigRNA', 'FileName_OrigRNA', args.root)
    metadata = relativePaths(metadata, 'ER', 'PathName_OrigER', 'FileName_OrigER', args.root)
    metadata = relativePaths(metadata, 'AGP', 'PathName_OrigAGP', 'FileName_OrigAGP', args.root)
    metadata = relativePaths(metadata, 'Mito', 'PathName_OrigMito', 'FileName_OrigMito', args.root)
    metadata = relativePaths(metadata, 'DNA', 'PathName_OrigDNA', 'FileName_OrigDNA', args.root)
    print metadata.info()
    # Merge two frames: metadata + plateMaps to attach treatment info to each image
    metadata = pd.merge(metadata, plateMaps.data, 
                        left_on=["Compound_Plate_Map_Name", "Metadata_Well"], 
                        right_on=["plate_map_name", "well_position"], how="left")
    print metadata.info()

if __name__ == "__main__":
    run()
