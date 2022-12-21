# acppred

By Frederico Schmitt Kremer

A tool to predict anticancer peptides

## Setup

$ make setup: either creates or updates the environment

# ACPPred Folder:

acppred/__ init __ .py: imports python scripts from this folder

acppred/models.py: creates class models where methods are defined

acppred/server.py: creates a flask app and links routes to reference .html files

acppred/train.py: sets up argument_parser arguments and stablishes tool default files. Import model and applies Random Forest estimator as default machine learning training algorithm

acppred/utils.py: holds aminoacids one letter code

# Data Folder

data/models: contains model.pickle archive

data/raw: contains positive.txt and negative.txt. Files that were used as training data for this platform

# Test Folder

tests/test_model.py: contains the methods used for testing the platform, which will be executed by pytest while testing

