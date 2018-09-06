#!/usr/bin/env python3

#import argparse
#import os.path
import re,os
import pandas as pd
from bs4 import BeautifulSoup
import requests, zipfile, io


def normalize_bbbc021_metadata(context):

    outpathI=context.obj["config"]["paths"]["images"]+"/"
    outpathM=context.obj["config"]["paths"]["metadata"]+"/"
    print(outpathI,outpathM)
    url = "https://data.broadinstitute.org/bbbc/BBBC021/"
    mbyte=1024*1024
    
    html = requests.get(url).text
    soup = BeautifulSoup(html,"lxml")
#    i=0
    for name in soup.findAll("a", href=True):
        zipurl = name["href"]
        if (zipurl.endswith(".zip")):
            print(zipurl.split("/")[-1])
            outfname = outpathI+ zipurl.split("/")[-1]
            r = requests.get(url+zipurl, stream=True)
            if( r.status_code == requests.codes.ok ) :
                fsize = int(r.headers["content-length"])
                print("Downloading %s (%sMb)" % ( outfname, fsize/mbyte ))
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(outpathI)
#                os.remove(outfname)   
        elif (zipurl.endswith("image.csv")):   
            print(zipurl.split("/")[-1])
            outfname2 = outpathM+ zipurl.split("/")[-1]
            print(outfname2)
            r = requests.get(url+zipurl, stream=True)
            if( r.status_code == requests.codes.ok ) :
                fsize = int(r.headers["content-length"])
                print("Downloading %s (%sMb)" % ( outfname2, fsize/mbyte ))
                 
                with open(outfname2, "wb") as fd:
                    for chunk in r.iter_content(chunk_size=1024): # chuck size can be larger
                        if chunk: # ignore keep-alive requests
                            fd.write(chunk)
                    fd.close()
    
    
    bbbc021 = pd.read_csv(outfname2)
    normalized = pd.DataFrame(columns=[
        "Metadata_Plate", "Metadata_Well", "Metadata_Site", "Plate_Map_Name",
        "DNA", "Tubulin", "Actin", "Replicate", "Compound_Concentration"
    ])
    
    
    def join(path_series, filename_series):
        return path_series + os.sep + filename_series
    
    
    normalized.Metadata_Plate = bbbc021.Image_Metadata_Plate_DAPI
    normalized.Metadata_Well = bbbc021.Image_Metadata_Well_DAPI
    normalized.DNA = join(bbbc021.Image_Metadata_Plate_DAPI,
                          bbbc021.Image_FileName_DAPI)
    normalized.Tubulin = join(bbbc021.Image_Metadata_Plate_DAPI,
                              bbbc021.Image_FileName_Tubulin)
    normalized.Actin = join(bbbc021.Image_Metadata_Plate_DAPI,
                            bbbc021.Image_FileName_Actin)
    normalized.Replicate = bbbc021.Replicate
    normalized.Metadata_Site = bbbc021.Image_FileName_DAPI.apply(
        lambda name: re.search(r"_(s\d+)_", name).group(1))
    normalized.Compound_Concentration = (
        bbbc021.Image_Metadata_Compound + "_" +
        bbbc021.Image_Metadata_Concentration.apply(str))
    
#    normalized.to_csv(outpathM+"NormalizedMetadata",sep=",", index=False)
    
    return