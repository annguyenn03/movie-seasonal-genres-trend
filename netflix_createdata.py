import sys, string, re
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types
import inflect

#IMDb dataset - Basic info for movies and shows
title_basics_schema = types.StructType([
    types.StructField('tconst', types.StringType()),
    types.StructField('titleType', types.StringType()),
    types.StructField('primaryTitle', types.StringType()),
    types.StructField('originalTitle', types.StringType()),
    types.StructField('isAdult', types.StringType()),
    types.StructField('startYear', types.StringType()),
    types.StructField('endYear', types.StringType()),
    types.StructField('runtimeMinutes', types.IntegerType()),
    types.StructField('genres', types.StringType())
])

#IMDb dataset - Ratings info for movies and shows
title_ratings_schema = types.StructType([
    types.StructField('tconst', types.StringType()),
    types.StructField('averageRating', types.DoubleType()),
    types.StructField('numVotes', types.IntegerType())
])

#Netflix dataset - Movies and Shows on Netflix
netflix_schema = types.StructType([
    types.StructField('show_id', types.StringType()),
    types.StructField('type', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('director', types.StringType()),
    types.StructField('cast', types.StringType()),
    types.StructField('country', types.StringType()),
    types.StructField('date_added', types.StringType()),
    types.StructField('release_year', types.IntegerType()),
    types.StructField('rating', types.StringType()),
    types.StructField('duration', types.StringType()),
    types.StructField('listed_in', types.StringType()),
    types.StructField('description', types.StringType()),
])

#Some special case for genres
special_case = {
    "Comedies" : "Comedy",
    "Romantic" : "Romance",
    'Dramas' : 'Drama'
}

#Function to Convert plurals to singleton
p = inflect.engine()
def to_singleton(genre_str):
    genre_str = genre_str.strip()#Remove any whitespace before and after the string

    #Remove extra words for TV Shows and Movies genres
    genre_str = genre_str.replace(" TV", "").replace(" Shows", "").replace(" Movies", "")

    #Convert to singleton
    genre_str = p.singular_noun(genre_str) or genre_str
    
    return genre_str

#Clean up some mismatch special case for genre
def to_special_case(genre):
    if genre in special_case:
        genre = special_case[genre]
    return genre
    


def main(in_directory, out_directory):
    #####From the Netflix Dataset#######
    #Read the Netflix data
    netflix = spark.read.option("sep", ",").option("header", "true").csv(in_directory + "/netflix_titles.csv", schema = netflix_schema)

    #Drop show_id because it has no usage
    #Drop duration because we have better info about duration in IMDb
    netflix = netflix.drop("show_id", "duration")

    #Rename rating to age_certification to avoid confusion with the numeric rating
    netflix = netflix.withColumnRenamed("rating", "ageCertification") 

    #Remove all rows with null value
    netflix = netflix.dropna()

    #Convert from plurals to singleton words in listed_in
    plurals_to_singleton = functions.udf(to_singleton, returnType=types.StringType())
    netflix = netflix.withColumn("listed_in", plurals_to_singleton(netflix['listed_in']))

    #####From the IMDb Dataset#######
    #Read basic infomation with the ratings
    title_basics = spark.read.option("sep", "\t").option("header", "true").csv(in_directory + "/title.basics.tsv.gz", schema = title_basics_schema)
    title_ratings = spark.read.option("sep", "\t").option("header", "true").csv(in_directory + "/title.ratings.tsv.gz", schema = title_ratings_schema)
    
    #Join basic info and ratings
    title = title_basics.join(title_ratings, on = "tconst")
    title = title.withColumnRenamed("primaryTitle", "title")
    title = title.withColumnRenamed("startYear", "release_year")

    #Remove all rows with null value
    title = title.dropna()

    # #Plurals to Singleton (In case there are mismatchs in the dataset)
    # title = title.withColumn("genres", plurals_to_singleton(title['genres']))

    #Join the dataset from IMDb and Netflix
    data = netflix.join(title, on=["title", 'release_year'])

    #####From the Combined Dataset (Main Dataset to work with)#######
    #Separate Genres for each movie/ show into array
    data = data.withColumn("genres", functions.split(data["genres"], ","))
    data = data.withColumn("listed_in", functions.split(data["listed_in"], r"\s*(?:,|&)\s*"))
    # data = data.withColumn("listed_in", functions.split(data["listed_in"], ","))

    #Trim space inside each element of the array in listed_in
    data = data.withColumn("listed_in", functions.transform(data["listed_in"], lambda x: functions.trim(x)))

    #Combine genres from IMDb and Netflix
    data = data.withColumn("genres", functions.array_union(data['listed_in'], data['genres']))
    data = data.drop("listed_in")

    #Rearrange order of the column (easier to look at)
    #Didn't choose endYear, description, director, cast because we will not work with it
    data = data.select('tconst', 'type', 'titleType', 'title', 'originalTitle', 'date_added', 'release_year',
    'genres', 'runtimeMinutes', 'averageRating', 'numVotes', 'ageCertification', 'isAdult', 'country')

    #Calculate the populartityScore
    score = data.groupBy().agg(functions.median("numVotes").alias('m'), functions.mean("averageRating").alias('C'))
    score_val = score.first()
    m = score_val['m']
    C = score_val['C']
    v = data['numVotes']
    R = data['averageRating']
    data = data.withColumn('popularityScore', (v / (v + m)) * R + (m / (v + m)) * C)

    #Convert to datetime format
    data = data.withColumn("date_added", functions.to_date(data['date_added'], format = 'MMMM d, yyyy'))

    #Add month column based on date_added
    data = data.withColumn("month_added", functions.month(data['date_added']))

    #Add season column based on month_added
    season_map = functions.when(
        data['month_added'].isin(1, 2, 3), "Spring").when(
            data['month_added'].isin(4, 5, 6), "Summer").when(
                data['month_added'].isin(7, 8, 9), "Fall").when(
                    data['month_added'].isin(10, 11, 12), "Winter")
    data = data.withColumn("season", season_map)

    #Explode to create each movie/show row with only one genre
    data_expanded = data.select('*', functions.explode("genres").alias('genre'))

    #Drop the array of genres (base genres) because we have already exploded each row with a single genre
    data_expanded = data_expanded.drop('genres')

    #Clean up some special genre cases that inflect can't convert to singleton
    convert_special_case = functions.udf(to_special_case, returnType=types.StringType())
    data_expanded = data_expanded.withColumn('genre', convert_special_case(data_expanded['genre']))

    #Write the result to csv file to do analysis
    data_expanded.write.csv(out_directory + "_cleaned", header = True, mode = 'overwrite')

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    spark = SparkSession.builder.appName('Netflix Movie and Shows Data').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)

