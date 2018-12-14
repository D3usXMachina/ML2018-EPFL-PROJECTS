================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~README~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
authors:                            Patrick Ley, Joel Fischer
date:                               29.10.2018
email: 								patrick.ley@epfl.ch, joel.fischer@epfl.ch
--------------------------------------------------------------------------------
Kaggle team name:                   0punctuation
--------------------------------------------------------------------------------
This team only consists of two members, which was exceptionally accepted for the
first project.
================================================================================
This folder contains the essential files used for the completion of the first
project in the 2018 machine learning course at EPFL by our group. To reproduce
the obtained predictions (which are stored in "submission.csv") run "run.py".
All the python functions used in "run.py" are either implemented in
"proj1_helpers.py" (which was provided and has not been modified) or in
"implementations.py". Detailed documentation of the implemented functions in
"implementations.py" is provided within the file.
The used methodology and the obtained results are discussed in "report.pdf".
================================================================================
Contents of this folder:

-README.txt             This document. Provides an overview of the content of
                        the submitted files.

-run.py                 The python script used to produce the submitted
                        predictions. (requires "proj1_helpers.py" and
                        "implementations.py").

-implementations.py     Python file containing the implementations of the
                        requested functions. All other functions required in
                        "run.py" as well some other functions that were used for
                        testing are also implemented in this file. (requires numpy)

-proj1_helpers.py       This python file was provided and contains functions to read
                        and write .csv files. (requires csv)

-report.pdf             Pdf containing a report on the methodology used and
                        and work done by our group.

-Submission.csv         A .csv file containing the predictions produced by the
                        python script "run.py" which were submitted on kaggle.

-select-model.ipynb     Jupyter notebook with some of the work done to find
                        the optimal hyper-parameters.

-plots.py               A slightly modified version of a file provided for lab04
                        which is used to create plots for cross-validation and
                        bias-variance-decomposition. (matplotlib.pyplot)

--------------------------------------------------------------------------------
Because of their size, the following files were not included in the submission.
They are required to run "run.py" and can be acquired from the GitHub repository
of this course.

-train.csv
-test.csv
================================================================================
