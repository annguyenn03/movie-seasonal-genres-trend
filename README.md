# cmpt-353-netflix-seasonal-genres-trend

# Dataset
There are 3 dataset were used in this project:
1. The Netflix dataset was downloaded from https://www.kaggle.com/datasets/shivamb/netflix-shows/data
2. The IMDb dataset was too large to upload to GitHub so you can download title.basics.tsv.gz and title.ratings.tsv.gz from https://datasets.imdbws.com/
Please make sure to have all dataset under the folder dataset

# How to Run
1. The data was cleaned and proccessed with Spark. After connected to Spark, you can run netflix_createdata.py with command:
spark-submit netflix_createdata.py dataset output
It will create a dataset under the folder output_cleaned

2. The analysis was read from the csv file from netflix_createdata.py and run with Pandas. It can be run command: python netflix_analysis.py
