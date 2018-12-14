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
# -Data preprocessing
# 	-> convertCSV2Surprise(infilepath,outfilepath=None)
# ==============================================================================
# TODO:
#   - nothing yet
# ==============================================================================


# ==============================================================================
# Data preprocessing
# ==============================================================================
# Some functions for reformatting and preprocessing user-item based data.
# ------------------------------------------------------------------------------

def convertCSV2Surprise(infilepath,outfilepath=None):
    """
    ----------------------------------------------------------------------------
    Creats a new file at <outfilepath> meeting the formatting requirements
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
    with the .csv ending changed to .dat (works only if the
    original file has .csv as suffix)

    Output:
    (none)
    ----------------------------------------------------------------------------
    """
    if ( outfilepath is None ) or ( outfilepath == infilepath ):
        outfilepath = infilepath.replace(".csv",".dat")
        if outfilepath[-4:] != ".dat":
            print("Please chose a valid path for the output file.")
            print("No output file was generated")
            return

    infile = open(infilepath,'rt')
    outfile = open(outfilepath,'wt')

    string = ''
    for line in infile:
        string = line.replace("r","",1)
        string = string.replace("_c"," ; ",1)
        string = string.replace(","," ; ",1)
        outfile.write(string)

    infile.close()
    outfile.close()

    return
# ------------------------------------------------------------------------------
