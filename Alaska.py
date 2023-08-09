import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


#pd.options.display.float_format = '{:.1f}'.format
filename = 'review-Alaska.json'

# Open file stream
df = pd.read_json(filename,lines=True, chunksize=200000, encoding_errors='ignore')


# ******DATA CLEANING / PREPROCESS*************************************************************

# remove 'pics' and 'resp' attributes, drop complete duplicates, and convert unix date to Y-M-D format
reviews = pd.DataFrame()
for chunk in df:
  chunk.drop(columns = ['pics', 'resp'], axis = 1, inplace = True)
  chunk.drop_duplicates(keep = 'last', inplace = True)
  chunk['date'] = pd.to_datetime(chunk['time'], unit = 'ms', origin = 'unix')
  chunk['mod_date'] = chunk['date'].dt.strftime('%Y-%m-%d')
  reviews = pd.concat([reviews, chunk]) 



# *****REVIEW CENTRIC FEATURES*************************************************************

# 1. Avg Word Count per Review
reviews['text'] = reviews['text'].astype(str)
reviews['totalwords'] = reviews['text'].str.count(' ') + 1
#data_top = reviews.head()
#print(data_top)

""" !!!!!!!GRAPH!!!!!!!!!
# is there a correlation between rating and word count? no
med_rating = reviews.groupby('rating').median('totalwords')
print(med_rating)
""" 

# get .25 and .75 quantiles for word counts to set threshold 
#print(reviews['totalwords'].quantile([0.25, 0.75, .5]))



# *****REVIEWER CENTRIC*************************************************************


# Avg reviews by user / day
reviews['num_reviewsPerUser'] = reviews['rating'].groupby(reviews['user_id']).transform('count')
reviews['countDays'] = reviews['mod_date'].groupby(reviews['user_id']).transform('nunique')
reviews['avgPerDay_User'] = reviews['num_reviewsPerUser'] / reviews['countDays']
df3 = reviews.sort_values('avgPerDay_User', ascending=False)
#print(df3.head(50))

# 1a. Boxplot for average reviews/day by user 
reviews.boxplot(column = 'avgPerDay_User').plot()
plt.title('Average # of Reviews per Day by User')
plt.show()



one_user = reviews.query('user_id == 103183143751502462976.0')['name']
#print("one_user " , one_user) 


# 2. Identify potentially fake names (e.g., Jon Do, Jane Do, all numeric) that have 5 star reviews
#reviews["user_id"] = reviews["user_id"].astype(str)
reviews['rating'] = pd.to_numeric(reviews['rating'], errors='coerce').fillna(0).astype(int)
reviews['fakeNames'] = np.where(((reviews['name'].str.contains('^john doe$|^john do$|^jon doe$|^jon do$|^jane doe$|^jane do$',case=False)) | reviews['name'].str.isnumeric()) & (reviews['rating'] == 5),True,False)
reviews['fakeNames'] = reviews['fakeNames'].astype(int)
df4 = reviews.sort_values('fakeNames', ascending=False)
#print(df4.head(50))

fakenamebyrating = reviews.groupby('rating').sum('fakeNames')
#print(fakenamebyrating)



# *****BUSINESS CENTRIC*************************************************************
# search for user_id appearing more than once per business (with 4 or 5 stars)
reviews['userIdCount'] = reviews.groupby(['gmap_id', 'user_id'])['user_id'].transform('count')
df5 = reviews.sort_values('userIdCount', ascending=False)
#print(df3.head(50))


""" in progress - outliers in # of reviews in time series by business id

reviews['Reviews_stdBus'] = reviews.groupby(['gmap_id', 'mod_date'])['rating'].transform('count')
print(reviews.head())
"""


# *****OTHER*************************************************************
"""
# to find datatype
datatypes = reviews.dtypes
#print("type: " ,datatypes)


# add columns for word ranges 0-5 and >5 words - COMMENTED OUT SO "WORD_RANGE" COLUMN DOESN'T EXIST
reviews['word_range'] = ["0-5" if x <=5 else ">5" for x in reviews['totalwords']]
df3 = reviews.groupby(['word_range'])['rating'].mean()
#print(df3)

#correlation coefficient
corr = reviews.word_range.corr(reviews['rating'])
#print("corr # words to rating is ", corr)

corr = num_reviews.rating.corr(num_reviews['Avg_per_Day'])
#print(corr)
"""

# new dataframe with dropped columns:
df2 = reviews.drop(columns = ['user_id','name','time','text','gmap_id','date','mod_date'], axis = 1)

