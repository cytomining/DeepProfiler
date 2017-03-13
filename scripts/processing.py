import pickle as pickle
import data.dataset as ds
import data.image_statistics as ists
import data.compression as cmpr

## The following functions are compatible with utils.Parallel
## These functions only accept one parameter with the args needed for computation
## Expected args: [plate, params] where plate is a dataframe, and params a dictionary

## STEP 02
## Computation of intensity stats per plate
## Required params: bits, channels, down_scale_factor, median_filter_size, output_dir, 
##                  image_dir, label_field
def intensityStats(args):
    # Load input parameters
    plate, params = args
    plateName = plate.data["Metadata_Plate"].iloc[0]

    # Create Dataset object
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dataset = ds.Dataset(plate, params["label_field"], params["channels"], params["image_dir"], keyGen)

    # Prepare ImageStatistics object
    hist = ists.ImageStatistics(params["bits"], params["channels"], params["down_scale_factor"], 
                                params["median_filter_size"], name=plateName)
    hist.expected = dataset.numberOfRecords("all")

    # Run the intensity computation
    dataset.scan(hist.processImage, frame="all")

    # Retrieve and store results
    stats = hist.computeStats()
    outfile = params["output_dir"] + plateName + ".pkl"
    with open(outfile,"wb") as output:
        pickle.dump(stats, output)

    return

## STEP 03
## Compression of images in a batch
## Required params: images_dir, stats_dir, output_dir, channels, source_format, scaling_factor
##                  label_field, control_field, control_value
def compressBatch(args):
    # Load parameters
    plate, params = args

    # Dataset configuration
    statsfile = params["stats_dir"] + plate.data.iloc[0]["Metadata_Plate"] + ".pkl"
    stats = pickle.load( open(statsfile, "rb") )
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dataset = ds.Dataset(plate, params["label_field"], params["channels"], params["images_dir"], keyGen)

    # Configure compression object
    compress = cmpr.Compress(stats, params["channels"], params["output_dir"])
    compress.setFormats(sourceFormat=params["source_format"], targetFormat="png")
    compress.setScalingFactor(params["scaling_factor"])
    compress.recomputePercentile(0.0001, side="lower")
    compress.recomputePercentile(0.9999, side="upper")
    compress.expected = dataset.numberOfRecords("all")

    # Setup control samples filter (for computing control illumination statistics)
    compress.setControlSamplesFilter(lambda x: x[params["control_field"]]==params["control_value"])

    # Run compression
    dataset.scan(compress.processImage, frame="all")

    # Retrieve and store results
    new_stats = compress.getUpdatedStats()
    with open(statsfile,"wb") as output:
        pickle.dump(new_stats, output)

    return
