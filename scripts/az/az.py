    
    ## Creation of training and validation partitions
    ### SPECIFIC TO AZ DATASET. Consider moving to another module

    def splitByReplicate(self, records):
        # One replicate for validation, two for training
        self.train = records[replicate(records,'1,2')].copy()
        self.val = records[replicate(records,'3')].copy()
        self.categories = self.train['Category'].unique()
        self.labels = dict([(self.categories[i],i) for i in range(len(self.categories))])
        self.train['Label'] = self.train['Category']
        self.train = self.train.replace({'Label':self.labels})
        self.val['Label'] = self.val['Category']
        self.val = self.val.replace({'Label':self.labels})

    def makeMonastrolSplit(self):
        self.data['Category'] = self.data['Image_Metadata_Compound']
        mask1 = self.data['Category'] == 'DMSO'
        mask2 = self.data['Category'] == 'monastrol'
        records = pd.concat( [self.data[mask1], self.data[mask2]] )
        self.splitByReplicate(records)
        print self.train.info()

    def makeMoASplit(self, moaGroundTruthFile):
        moa_data = pd.read_csv(moaGroundTruthFile)
        moa_data.columns = ['Image_Metadata_Compound','Image_Metadata_Concentration','MOA']
        self.data = self.data.merge(moa_data)
        self.data['Category'] = self.data['Image_Metadata_Compound']
        self.splitByReplicate(self.data)
        print self.train.info()

###########################
## Finding pixel statistics
def computeMeanAndStd(metadataFilename, datasetDir, outputFile):
    metadata = meta.ImageMetadata(metadataFilename)
    metadata.makeMonastrolSplit()
    dataset = ds.Dataset(metadata, dataRoot=datasetDir)
    hist = Histogram(16, 3)
    hist.expected = dataset.numberOfRecords('val')
    dataset.scan(hist.processImage, frame='val')
    stats = hist.computeStats()
    with open(outputFile, 'wb') as output:
        pickle.dump(stats, output, protocol=pickle.HIGHEST_PROTOCOL)
    return hist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Filename of the CSV metadata file")
    parser.add_argument("datadir", help="Directory containing images listed in the CSV")
    parser.add_argument("outfile", help="Filename of the resulting pickle file with the mean an std")
    args = parser.parse_args()
    computeMeanAndStd(args.metadata, args.datadir, args.outfile)
