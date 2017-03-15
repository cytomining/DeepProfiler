import pickle as pickle
import data.utils as utils
import data.dataset as ds
import data.image_statistics as ists
import data.compression as cmpr


## Basic functionalities

def illum_stats_filename(output_dir, plate_name):
    return output_dir + "/" + plate_name + "/intensities/" + plate_name + ".pkl"

def png_dir(output_dir, plate_name):
    return output_dir + "/" + plate_name + "/pngs/"

## The following functions are compatible with utils.Parallel
## These functions only accept one parameter with the args needed for computation
## Expected args: [plate, config] where plate is a dataframe, and config a dictionary
## The metadata is assumed to be produced by CellProfiler and Cytominer, hence the field names
## Parameterizing these field names might be required in the future

## STEP 02
## Computation of intensity stats per plate
def intensityStats(args):
    # Load input parameters
    plate, config = args
    plateName = plate.data["Metadata_Plate"].iloc[0]

    # Create Dataset object
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dataset = ds.Dataset(plate, config["metadata"]["label_field"], 
                                config["original_images"]["channels"], 
                                config["original_images"]["path"], 
                                keyGen)

    # Prepare ImageStatistics object
    hist = ists.ImageStatistics(config["original_images"]["bits"], 
                                config["original_images"]["channels"], 
                                config["illumination_correction"]["down_scale_factor"], 
                                config["illumination_correction"]["median_filter_size"], 
                                name=plateName)
    hist.expected = dataset.numberOfRecords("all")

    # Run the intensity computation
    dataset.scan(hist.processImage, frame="all")

    # Retrieve and store results
    stats = hist.computeStats()
    outfile = illum_stats_filename(config["compression"]["output_dir"], plateName) 
    utils.check_path(outfile)
    with open(outfile,"wb") as output:
        pickle.dump(stats, output)

    return

## STEP 03
## Compression of images in a batch
def compressBatch(args):
    # Load parameters
    plate, config = args
    plate_name = plate.data.iloc[0]["Metadata_Plate"]

    # Dataset configuration
    statsfile = illum_stats_filename(config["compression"]["output_dir"], plate_name)
    stats = pickle.load( open(statsfile, "rb") )
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dataset = ds.Dataset(plate, 
                         config["metadata"]["label_field"], 
                         config["original_images"]["channels"], 
                         config["original_images"]["path"], 
                         keyGen)

    # Configure compression object
    plate_output_dir = png_dir(config["compression"]["output_dir"], plate_name)
    compress = cmpr.Compress(stats, config["original_images"]["channels"], plate_output_dir)
    compress.setFormats(sourceFormat=config["original_images"]["file_format"], targetFormat="png")
    compress.setScalingFactor(config["compression"]["scaling_factor"])
    compress.recomputePercentile(0.0001, side="lower")
    compress.recomputePercentile(0.9999, side="upper")
    compress.expected = dataset.numberOfRecords("all")

    # Setup control samples filter (for computing control illumination statistics)
    filter_func = lambda x: x[config["metadata"]["control_field"]]==config["metadata"]["control_value"]
    compress.setControlSamplesFilter(filter_func)

    # Run compression
    dataset.scan(compress.processImage, frame="all")

    # Retrieve and store results
    new_stats = compress.getUpdatedStats()
    with open(statsfile,"wb") as output:
        pickle.dump(new_stats, output)

    return
