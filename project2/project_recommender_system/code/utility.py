# -*- coding: utf-8 -*-
# ==============================================================================
# utility.py
# ------------------------------------------------------------------------------
# authors:                             Patrick Ley, Joel Fischer
# date:                                14.12.2018
# ==============================================================================
# This code collection contains some functions used for our realsation of
# project2: recommender system.
# ==============================================================================
# - File processing and Import/Export
#   -> loadData2df(infilepath,skiplines=1)
#   -> loadData2ds(source,skiplines=1)
#   -> generatePredictions(algo,pred_df,verbose)
#   -> exportPredictions(outfilepath,pred_df)
#
# - Deprecated methods
#   ((DEPRECATED) -> convertCSV2Surprise(infilepath,outfilepath=None) )
#   ((DEPRECATED) -> properCSV(infilepath,outfilepath=None,seperator=None) )
# ==============================================================================
# TODO:
#   - nothing yet
#   -
# ==============================================================================

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import Reader

# ==============================================================================
# File processing and Import/Export
# ==============================================================================
# Some functions for reformatting and importing/exporting user-item-rating data.
# ------------------------------------------------------------------------------

def loadData2df(infilepath,skiplines=1):
    """
    ============================================================================
    Creates a pandas.DataFrame from a data file with format
    ("r<user>_c<item>,<rating>"). The first line is skipped.
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    Input:

    - infilepath 	(string,valid path);
                    filepath to the original .csv file;

    - skiplines     (positive integer);
                    number of lines to be skipped at start of file;
                    (default=1);

    Output:

    - df            (pandas DataFrame);
                    pandas data frame containing ratings, can be loaded by
                    scikit-surprise;
    ============================================================================
    """

    def readLine(line):
        userId, rating = line.split(',')
        userId, itemId = userId.split("_c")
        userId = userId.replace("r", "")
        return int(userId), int(itemId), float(rating)

    skiplines = max(0,skiplines)

    infile = open(infilepath,"rt")
    lines = infile.read().splitlines()[skiplines:]
    nratings = len(lines)
    data = {\
    "userId": [0]*nratings,\
    "itemId": [0]*nratings,\
    "rating": [3.0]*nratings\
    }
    uid = 0
    iid = 0
    r = 3.5
    for ln, line in enumerate(lines):
        uid,iid,r = readLine(line)
        data["userId"][ln] = uid
        data["itemId"][ln] = iid
        data["rating"][ln] = r

    df = pd.DataFrame(data)
    infile.close()

    return df

# ------------------------------------------------------------------------------

def generatePredictions(algo,pred_df,verbosity=0):
    """
    ============================================================================
    Predict ratings .
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    Input:

    - outfilepath   (string,valid path);
                    filepath to the newly created file;

    - pred_df       (pandas.DataFrame);
                    DataFrame containing the predicted ratings

    - verbosity     (integer);
                    print out more information the higher the integer:
                    0 : no text output
                    1 : print (#rating,uid,iid,rating)
                    2 : also print verbose predict output

    Output: (nothing)
    ============================================================================
    """

    verb = (verbosity>1)
    for i in range(pred_df.shape[0]):
        pred_df.at[i,"rating"] = algo.predict(str(pred_df.at[i,"userId"])\
                                             ,str(pred_df.at[i,"itemId"]),verbose=verb).est
        if(verbosity>0):
            print(i,pred_df["userId"][i],pred_df["itemId"][i],pred_df["rating"][i])

    return

# ------------------------------------------------------------------------------

def addDateAndTime(outfilepath,add_date=False,add_time=False):

    if add_date or add_time:
        datestring = str(datetime.datetime.now())
        datestring = datestring.replace(" ","_")
        datestring = datestring.replace(":","")
        if not add_time:
            datestring = datestring[:-5]
        if not add_date:
            datestring = datestring[11:]
        outfilepath = outfilepath.replace(".","_"+a+".")

    return outfilepath

def exportPredictions(outfilepath,pred_df,add_date=False,add_time=False):
    """
    ============================================================================
    Export predictions from a pandas.DataFrame to a file in the format demanded
    by the automatic evaluation algorithm.
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    Input:

    - outfilepath   (string,valid path);
                    filepath to the newly created file;

    - pred_df       (pandas.DataFrame);
                    DataFrame containing the predicted ratings

    Output: (nothing)
    ============================================================================
    """
    outfilepath = addDateAndTime(outfilepath,add_date,add_time)

    outfile = open(outfilepath,"wt")
    outfile.write("Id,Prediction\n")
    for i, rating in pred_df.iterrows():
        outfile.write("r"+str(int(rating["userId"]))+"_c"+str(int(rating["itemId"]))+","+str(round(rating["rating"]))+"\n")

    outfile.close()

    return

def exportAlgoParameters(algo,outfilepath=None,add_date=False,add_time=False):

    if outfilepath is None:
        outfilepath = "algoPara.csv"
        outfilepath = addDateAndTime(outfilepath,True,True)
    else:
        outfilepath = addDateAndTime(outfilepath,add_date,add_time)

    outfile = open(outfilepath,"w")

    attlist = dir(algo)
    for att in attlist:
        if att[0] != "_" and not callable(getattr(algo,att)):
            outfile.write(att+":"+str(type(getattr(algo,att)))+"\n")
            outfile.write(str(getattr(algo,att)))
            outfile.write("\n")

    outfile.close()

    return

# ------------------------------------------------------------------------------

def loadData2ds(source=None,skiplines=1):
    """
    ============================================================================
    Load rating data from panda.DataFrame or an external file to surprise.Dataset.
    ----------------------------------------------------------------------------
    note: it is recommended to first load the data to a panda.DataFrame and then
    convert it.
    ----------------------------------------------------------------------------
    Input:

    - source 	    (string,valid path)/(panda.DataFrame);
                    filepath to the original .csv file/ DataFrame containing ratings;

    - skiplines     (positive integer);
                    number of lines to be skipped at start of file;
                    only used when directly loading from file;
                    (default=1);

    Output:

    - df            (surprise.Dataset);
                    data set containing ratings
    ============================================================================
    """

    reader = Reader(rating_scale=(1,5))

    if type(source) is pd.DataFrame:
        df = source
    elif type(source) is str:
        df = load2df(source,skiplines)
    else:
        print("No valid source. Use either a panda.DataFrame a valid filepath.")
        return

    ds = Dataset.load_from_df(source,reader)

    return ds

# ==============================================================================
# Deprecated Methods
# ==============================================================================
# Some functions that aren't updated anymore/ are rendered superfluent but that
# are still kept for the sake of backwards compatibility
# ------------------------------------------------------------------------------

def properCSV(infilepath,outfilepath=None,seperator=None):
    """
    DEPRECATED METHOD, use loadData and pandas.DataFrame.to_csv instead
    ============================================================================
    CSV in .csv stands for comma seperated values, so why not seperate values
    with commas?
    ----------------------------------------------------------------------------
    Creats a new .csv file at <outfilepath> meeting the formatting requirements
    ("<user> , <item> , <rating>") for the "Reader" class in the "surprise"
    library from a .csv file at <infilepath> with the format
    ("r<user>_c<item>,<rating>").

    note:-if there already exists a file at <outfilepath> it will be overwritten
         -if the original file does not have the correct format the file might
          not be converted properly
    ----------------------------------------------------------------------------
    Input:

    - infilepath 	(string,valid path);
                    filepath to the original .csv file;

    - outfilepath 	(string,valid path);
                    filepath to the newly created file;
                    if either the same name as the input file or no value at all
                    is passed, the output filepath will be the input filepath
                    with a -wc suffix (works only if the
                    original file has .csv as filename extension)

    - seperator     (string);
                    seperator of choice (default = ",");

    Output: (nothing)
    ============================================================================
    """
    # check for output file path
    if ( outfilepath is None ) or ( outfilepath == infilepath ):
        outfilepath = infilepath.replace(".csv","-wc.csv")
        print(infilepath)
        print(outfilepath)
        if outfilepath[-7:] != "-wc.csv":
            print("Please chose an input file with the .csv filename extension or pass an explicit output filepath.")
            print("No output file was generated")
            return
    # ceck for costum seperator
    if seperator is None:
        seperator = ","

    infile = open(infilepath,'rt')
    outfile = open(outfilepath,'wt')

    string = ''
    for linenb, line in enumerate(infile):
        if linenb > 0:
            string = line.replace("r","",1)
            string = string.replace("_c",seperator,1)
            string = string.replace(",",seperator,1)
            outfile.write(string)
        else:
            outfile.write("userId"+seperator+"itemId"+seperator+"rating\n")

    infile.close()
    outfile.close()

    return

# ------------------------------------------------------------------------------

def convertCSV2Surprise(infilepath,outfilepath=None):
    """
    DEPRECATED METHOD, use loadData and pandas.DataFrame.to_csv instead
    ============================================================================
    CSV in .csv stands for comma seperated values, so why not seperate values
    with commas?
    ----------------------------------------------------------------------------
    Creats a new .csv file at <outfilepath> meeting the formatting requirements
    ("<user> ; <item> ; <rating>") for the "Reader" class in the "surprise"
    library from a .csv file at <infilepath> with the format
    ("r<user>_c<item>,<rating>").

    note:-if there already exists a file at <outfilepath> it will be overwritten
         -if the original file does not have the correct format the file might
          not be converted properly
    ----------------------------------------------------------------------------
    Input:

    - infilepath 	(string,valid path);
                    filepath to the original .csv file;

    - outfilepath 	(string,valid path);
                    filepath to the newly created file;
                    if either the same name as the input file or no value at all
                    is passed, the output filepath will be the input filepath
                    with a -wc suffix (works only if the
                    original file has .csv as filename extension)

    Output: (nothing)
    ============================================================================
    """
    properCSV(infilepath,outfilepath,seperator=" ; ")
    return
