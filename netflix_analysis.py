# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns

# %%
#Read the file
folder_path = 'output_cleaned'
files = os.listdir(folder_path)
data_list = []

for file in files:
    if file.endswith(".csv"):
        path = os.path.join(folder_path, file)
        data = pd.read_csv(path)
        data_list.append(data)

data = pd.concat(data_list)

# %%
#Convert date_added to datetime
data['date_added'] = pd.to_datetime(data['date_added'])

# %%
#Question: Do some genres appear more often than other in some seasons
contigency = pd.crosstab(data['genre'], data['season'])
genre_freq_p = stats.chi2_contingency(contigency).pvalue
print(genre_freq_p)

#plot for visualization
contigency.plot(kind = 'bar', stacked = True)
plt.title("Genres distribution over years by seasons")
plt.xlabel("Genre")
plt.ylabel("Number of move/shows")
plt.savefig('season_distribution.svg')

# %%
#Group all data back - Not separate by genre each row anymore
grouped_data = data.groupby(['tconst', 'type', 'titleType', 'title', 'originalTitle', 'date_added',
                             'release_year', 'runtimeMinutes', 'averageRating',	'numVotes',
                             'ageCertification', 'isAdult', 'country',
                             'popularityScore', 'month_added', 'season']).aggregate({'genre': list}).reset_index()

# %%
#Question: Does month added affect the rating for each movie/show?
#Find the correlation coefficient
fit = stats.linregress(grouped_data['month_added'], grouped_data['averageRating'])
corr = fit.rvalue
print("The correlation coefficient between month added and rating is: ", corr)

#Prediction for the next rating with the given month
grouped_data['rating_pred'] = grouped_data['month_added'] * fit.slope + fit.intercept

#Plot for better visualization
plt.plot(grouped_data['month_added'], grouped_data['averageRating'], 'o')
plt.plot(grouped_data['month_added'], grouped_data['rating_pred'], 'r-')
plt.xlabel('Month Added')
plt.ylabel('Average Rating')
plt.title('Relationship between month added and average rating on Netflix')
plt.savefig('month_rating_corr.svg')

# %%
#Question: Has movie and show on Netflix popularity increased over time?
#Find the correlation coefficient
fit_popularity = stats.linregress(grouped_data['release_year'], grouped_data['popularityScore'])
corr_popularity = fit_popularity.rvalue
print("The correlation coefficient between month added and popularity score is: ", corr_popularity)

#Prediction for the next rating with the given month
grouped_data['popularity_pred'] = grouped_data['release_year'] * fit_popularity.slope + fit_popularity.intercept

#Plot for better visualization
plt.plot(grouped_data['release_year'], grouped_data['popularityScore'], 'o')
plt.plot(grouped_data['release_year'], grouped_data['popularity_pred'], 'r-')
plt.xlabel('Released Year')
plt.ylabel('Popularity Score')
plt.title('Relationship between year released and popularity score on Netflix')
plt.savefig('year_popularity_corr.svg')

# %%
#Question: Predict the popularity score given the runtimeMinutes, and release_year
X = grouped_data[['runtimeMinutes', 'numVotes', 'release_year']].values
y = grouped_data['popularityScore'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

print("The accuracy score for training set = ", model.score(X_train, y_train))
print("The accuracy score for validation set = ", model.score(X_valid, y_valid))
print("Feature Coefficient = ", model.feature_importances_)


