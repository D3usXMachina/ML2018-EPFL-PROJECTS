# -*- coding: utf-8 -*-
# ==============================================================================
# utility.py
# ------------------------------------------------------------------------------
# authors:                             Patrick Ley, Joel Fischer
# date:                                14.12.2018
# ==============================================================================
# This code collection contains some functions used for our realisation of
# project2: recommender system.
# ==============================================================================
# - File processing and Import
#   -> loadData2df(infilepath,skiplines=1)
#   -> loadData2ds(source,skiplines=1)
#   -> filterDf(df,min_ratings_user,min_ratings_item)
# - Predictions and Parameter export
#   -> generatePredictions(algo,pred_df,verbose)
#   -> addDateAndTime(filepath,add_date=True,add_time=True)
#   -> exportPredictions(pred_df,outfilepath=None,add_date=False,add_time=False)
#   -> exportAlgoParameters(algo,outfilepath=None,add_date=False,add_time=False)
#
# - Deprecated methods
#   ((DEPRECATED) -> convertCSV2Surprise(infilepath,outfilepath=None) )
#   ((DEPRECATED) -> properCSV(infilepath,outfilepath=None,seperator=None) )
# ==============================================================================
# TODO:
#   - add function to export gridseach/cross-validation data
#   - add function to plot gridsearch/cross-validation data
# ==============================================================================

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import Reader

# ==============================================================================
# File processing and Import
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

# ------------------------------------------------------------------------------

def filterDf(df,min_ratings_user,min_ratings_item):
    """
    ============================================================================
    Filter user-item-ratings dataframe by a minimum number of ratings
    requirement.
    ----------------------------------------------------------------------------
    Returns a new pandas.DataFrame where users who gave less than <min_ratings_user>
    ratings and items which recieved less than <min_ratings_item> are excluded.
    ----------------------------------------------------------------------------
    Input:

    - df                (pandas.DataFrame);
                        DataFrame containing ratings;

    - min_ratings_user  (integer);
                        minimum reuired ratings given by user;

    - min_ratings_item  (integer);
                        minimum reuired ratings recieved by item;

    Output:

    - df                (pandas.DataFrame);
                        filtered dataframe
    ============================================================================
    """

    reduced_df = df

    # filter on items
    item_rating_counts = reduced_df["itemId"].value_counts()
    item_rating_counts = item_rating_counts[item_rating_counts>=min_ratings_item]
    reduced_df = reduced_df[reduced_df["itemId"].isin(item_rating_counts.index)]

    # filter on users
    user_rating_counts = reduced_df["userId"].value_counts()
    user_rating_counts = user_rating_counts[user_rating_counts>=min_ratings_user]
    reduced_df = reduced_df[reduced_df["userId"].isin(user_rating_counts.index)]

    return reduced_df

# ==============================================================================
# Predictions and Parameter export
# ==============================================================================
# Some functions to automate predictions and algorithm parameter export.
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
    raw_id_type = type(algo.trainset.to_raw_iid(0))
    verb = (verbosity>1)
    iid = 0
    uid = 0
    for i in range(pred_df.shape[0]):
        uid = pred_df.at[i,"userId"]
        iid = pred_df.at[i,"itemId"]
        if raw_id_type is str:
            uid = str(uid)
            iid = str(iid)
        pred_df.at[i,"rating"] = algo.predict(uid,iid,verbose=verb).est
        if(verbosity>0):
            print(i,pred_df["userId"][i],pred_df["itemId"][i],pred_df["rating"][i])

    return

# ------------------------------------------------------------------------------

def addDateAndTime(filepath,add_date=True,add_time=True):
    """
    ============================================================================
    Add date and time to a filename.
    ----------------------------------------------------------------------------
    Adds the current date and time in the filepath before the filename extension
    in the format "_YYYY-MM-DD_HHMM".
    example:
    "../path/filename.extension" -> "../path/filename_<date>_<time>.extension"
    ----------------------------------------------------------------------------
    Input:

    - filepath      (string,valid path);
                    filepath to be modified;

    - add_date      (boolean);
                    add date to filename;
                    (default=False);

    - add_time      (boolean);
                    add time to filename;
                    (default=False);

    Output:

    - filepath_new  (string);
                    the modified filepath
    ============================================================================
    """

    filepath_new = filepath
    if add_date or add_time:
        datestring = str(datetime.datetime.now())[:16]
        datestring = datestring.replace(" ","_")
        datestring = datestring.replace(":","")
        if not add_time:
            datestring = datestring[:-5]
        if not add_date:
            datestring = datestring[11:]
        name, extension = filepath_new.rsplit(".",1)
        filepath_new = name + "_" + datestring + "." + extension

    return filepath_new

# ------------------------------------------------------------------------------

def exportPredictions(pred_df,outfilepath=None,add_date=False,add_time=False):
    """
    ============================================================================
    Export predictions from a pandas.DataFrame to a file in the format demanded
    by the automatic evaluation algorithm.
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    Input:

    - outfilepath   (string,valid path);
                    filepath to the newly created file;
                    (default="../data/submissions/submission_<datetime>.csv");

    - pred_df       (pandas.DataFrame);
                    DataFrame containing the predicted ratings;

    - add_date      (boolean);
                    add date to filename;
                    (default=False);

    - add_time      (boolean);
                    add time to filename;
                    (default=False);

    Output: (nothing)
    ============================================================================
    """

    if outfilepath is None:
        outfilepath_new = "../data/submissions/submission.csv"
        outfilepath_new = addDateAndTime(outfilepath_new,True,True)
    else:
        outfilepath_new = outfilepath
        outfilepath_new = addDateAndTime(outfilepath_new,add_date,add_time)

    outfile = open(outfilepath_new,"wt")
    outfile.write("Id,Prediction\n")
    for i, rating in pred_df.iterrows():
        outfile.write("r"+str(int(rating["userId"]))+"_c"+str(int(rating["itemId"]))+","+str(round(rating["rating"]))+"\n")

    outfile.close()

    return

# ------------------------------------------------------------------------------

def exportAlgoParameters(algo,outfilepath=None,add_date=False,add_time=False):
    """
    ============================================================================
    Export the parameters used by the algo.
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    Input:

    - pred_df       (surprise.BaseAlgo);
                    surprise algorithm used to make predictions;

    - outfilepath   (string,valid path);
                    filepath to the newly created file;
                    (default="../data/parameters/parameters_<datetime>.dat");

    - add_date      (boolean);
                    add date to filename;
                    (default=False);

    - add_time      (boolean);
                    add time to filename;
                    (default=False);

    Output: (nothing)
    ============================================================================
    """

    if outfilepath is None:
        outfilepath_new = "../data/parameters/parameters.dat"
        outfilepath_new = addDateAndTime(outfilepath_new,True,True)
    else:
        outfilepath_new = outfilepath
        outfilepath_new = addDateAndTime(outfilepath_new,add_date,add_time)

    outfile = open(outfilepath_new,"w")

    outfile.write("algo:"+str(getattr(algo,"__class__")))
    attlist = dir(algo)
    for att in attlist:
        if att[0] != "_" and not callable(getattr(algo,att)):
            outfile.write(att+":"+str(type(getattr(algo,att)))+"\n")
            outfile.write(str(getattr(algo,att)))
            outfile.write("\n")

    outfile.close()

    return

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
