import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
import seaborn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import seaborn as sns
 


#pd.options.display.float_format = '{:.1f}'.format
filename = 'review-Alaska.json'

# Open file stream
df = pd.read_json(filename,lines=True, chunksize=200000, encoding_errors='ignore')


# ******DATA CLEANING / PREPROCESS*************************************************************

# remove 'pics' and 'resp' attributes, drop complete duplicates, and convert unix date to Y-M-D format
reviews = pd.DataFrame()
for chunk in df:
  chunk.drop(columns = ['pics', 'resp'], axis = 1, inplace = True)
  chunk = chunk.dropna()
  chunk.drop_duplicates(keep = 'last', inplace = True)
  chunk['date'] = pd.to_datetime(chunk['time'], unit = 'ms', origin = 'unix')
  chunk['mod_date'] = chunk['date'].dt.strftime('%Y-%m-%d')
  reviews = pd.concat([reviews, chunk]) 



# *****REVIEW CENTRIC FEATURES*************************************************************

# 1. Avg Word Count per Review
reviews['text'] = reviews['text'].astype(str)
reviews['totalwords'] = reviews['text'].str.count(' ') + 1
# print(reviews.head())



# get .25 and .75 quantiles for word counts to set threshold 
#print(reviews['totalwords'].quantile([0.25, 0.75, .5]))



# *****REVIEWER CENTRIC*************************************************************


#Review Volume (Reviewer Centric)-------------------------------------

reviews['num_reviewsPerUser'] = reviews['rating'].groupby(reviews['user_id']).transform('count')
reviews['countDays'] = reviews['mod_date'].groupby(reviews['user_id']).transform('nunique')
reviews['avgPerDay_User'] = reviews['num_reviewsPerUser'] / reviews['countDays']

#print(reviews.head())


# number of unique user_ids with more than 3 avg reviews/day
result = reviews.loc[reviews['avgPerDay_User'] > 3, 'user_id'].nunique()
#print("number of users with more than 3 avg reviews/day: ",result)


# Quantiles
result5 = reviews.loc[(reviews['avgPerDay_User'] > 3)]
#print("num words - users with > avg 3 reviews per day: ",result5['totalwords'].quantile([0.25, 0.75, .5]))
# get .25 and .75 quantiles for word counts for users with > 3 reviews/day
#print("num words - all reviews: ",reviews['totalwords'].quantile([0.25, 0.75, .5]))



#Review Content: Mean Word Count (Reviewer Centric)-----------------------

# mean number of review words for each user
reviews['meanWords_User'] = reviews['totalwords'].groupby(reviews['user_id']).transform('mean')
df7 = reviews.sort_values('meanWords_User', ascending = False)
#print("mean words by user: ",df7[['user_id','name','meanWords_User']].head(50))

# metric for mean number of review words for each user (if less than 2 -> 1, else 1/mean)
reviews['meanWords_UserMetric'] = np.where(reviews['meanWords_User'] < 2, 1, 1/reviews['meanWords_User'])
#print(reviews[['user_id','name','meanWords_UserMetric']].head(50))

# check one user
one_user = reviews.query('user_id == 103183143751502462976.0')['name']
#print("one_user " , one_user) 


#Review Content: Key Word Count - Flag Overly Enthusiastic Reviews (Reviewer Centric)

"""
#count number of key words that appear in text of each review, based on key_words list
key_words = ['!','awesome','best','better','brilliant','delightful','elated','enjoyed','excellent','exceptional','extraordinary','fabulous','fantastic','first class','first-class','fun','glorious','great','happy','happiest','kind','kindest','love','loved','magical','marvelous','outstanding','overjoyed','perfect','quality','recommend','splendid','super','superb','superior','supreme','terrific','top notch','top-notch','treasure','unparalleled','value','valued','valuable','wonderful','worth','worthy']

reviews['keyWordCount'] = reviews['text'].str.count(fr"\b(?:{'|'.join(key_words)})\b")
print(reviews[['user_id','name','keyWordCount']].head())
reviews['keyWordSum_User'] = reviews['user_id'].groupby(reviews['key_words']).transform('sum')
reviews['wordSum_User'] = reviews['user_id'].groupby(reviews['totalwords']).transform('sum')
reviews['textContent'] = reviews['user_id'].groupby(reviews['totalwords']).transform('sum')

#reviews['textContent'] = reviews.groupby(reviews['user_id']).sum('keyWordCount') / reviews.groupby(reviews['user_id']).sum('totalwords')
#print(reviews.head())

"""

#Fake Names (Reviewer Centric)--------------------------------------------

# Identify potentially fake reviews via user names (e.g., Jon Do, Jane Do, A Google User, all numeric) that have 5 star reviews
# reviews["user_id"] = reviews["user_id"].astype(str)
reviews['rating'] = pd.to_numeric(reviews['rating'], errors='coerce').fillna(0).astype(int)
reviews['fakeNames'] = np.where(((reviews['name'].str.contains('^john doe$|^john do$|^jon doe$|^jon do$|^jane doe$|^jane do$|^a google user$',case=False)) | reviews['name'].str.isnumeric()) & (reviews['rating'] == 5),True,False)
reviews['fakeNames'] = reviews['fakeNames'].astype(int)
df4 = reviews.sort_values('fakeNames', ascending=False) #sort to view results
# print(df4.head(50))

# total fake names per rating
fakenamebyrating = reviews.groupby('rating').sum('fakeNames')
# print(fakenamebyrating)

# number of user_ids with >5 reviews/day and fake name flag
result4 = reviews.loc[(reviews['avgPerDay_User'] > 5) & (reviews['fakeNames'] == 1),'user_id'].nunique()
# print("user with avg reviews/day > 5 and fake name flag: ",result4)

# Average rating per user - for Fake Name Metric
reviews['meanRating_User'] = reviews['rating'].groupby(reviews['user_id']).transform('mean')
# print("meanReviews_User ",reviews[['user_id','name','meanReviews_User']].head(50))

reviews['FakeNameMetric'] = np.where((reviews['meanRating_User'] > 3) & (reviews['fakeNames'] == 1), 1, 0)
#sortFN = reviews.sort_values('FakeNameMetric', ascending=False)
#print(sortFN[['user_id','name','meanRating_User','fakeNames','FakeNameMetric']].head(50))

# User Names per ID (Reviewer Centric)----------------------------------------


# number of unique names under user id
reviews['numUserNames_User'] = reviews['name'].groupby(reviews['user_id']).transform('nunique')
df5 = reviews.sort_values('numUserNames_User', ascending = False)
#print(df5.head(50))


# per rating, how many userids w/ more than one name?
reviews['nameFlag'] = np.where(reviews['numUserNames_User'] > 1, True, False).astype(int) #flag if user has more than 1 name
df6 = reviews.sort_values('nameFlag', ascending = False)
#print(df6.head(50))

# add distinct count of user_ids with more than one name by rating


result1 = reviews.groupby('nameFlag')['user_id'].nunique()
#print(result1)


# Metric: user_ids with mean rewiew rating > 3 and more than 1 user name = 1, otherwise 0
reviews['NameCountMetric'] = np.where((reviews['meanRating_User'] > 3) & (reviews['nameFlag'] == 1), 1, 0)

# Total Metrics
reviews['FakeMetric_User'] = reviews['meanWords_User'] 
#print(reviews.head())



# *****BUSINESS CENTRIC*************************************************************

"""
# search for user_id appearing more than once per business (with 4 or 5 stars)
reviews['userIdCount'] = reviews.groupby(['gmap_id', 'user_id'])['user_id'].transform('count')
df8 = reviews.sort_values('userIdCount', ascending=False)
#print(df8.head(50))
"""

"""
# boolean - true if business has more than 1 review
reviews['numReviews_bus'] = reviews['rating'].groupby(reviews['gmap_id']).transform('count')
#print(reviews.head(50))
"""
    
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
df9 = reviews.groupby(['word_range'])['rating'].mean()
#print(df9)
"""

"""
#correlation coefficient
corr = reviews.FakeNameMetric.corr(reviews['rating'])
print("rating to FakeName correlation is ", corr)


corr = reviews.FakeMetric_User.corr(reviews['meanRating_User'])
print(corr)
"""


df2 = reviews.drop(columns = ['time','text','gmap_id','date','mod_date','nameFlag','meanWords_UserMetric','meanRating_User','FakeNameMetric','numUserNames_User','NameCountMetric','countDays','num_reviewsPerUser','totalwords','rating','fakeNames','FakeMetric_User'], axis = 1)


df2 = df2.drop_duplicates(subset=['user_id'], keep="first")
print("drop dup ids" , df2.head(100))


#print("df2 col: ",list(df2.columns))



# *****k-Means*************************************************************


dfTrim = df2.drop(columns =['name','user_id'], axis = 1)
#dfs=dfTrim
#dfs=(dfTrim-dfTrim.min())/(dfTrim.max()-dfTrim.min())
df_cluster = dfTrim[['avgPerDay_User', 'meanWords_User']]
kmeans = KMeans(n_clusters=2)
kmeans.fit(df_cluster)



kmeans.labels_  #added
df2['cluster']= kmeans.labels_
clusterCount = df2.groupby(['cluster'])['cluster'].count()
print(clusterCount)

numIds = df2.user_id.nunique()
print("user id count ",numIds)

"""
dfBar = df2.drop(columns=['avgPerDay_User','meanWords_User'], axis=1)
df2.groupby(["cluster"])["cluster"].count().plot(kind="bar", stacked=True)
"""


plt.scatter(df_cluster['avgPerDay_User'], df_cluster['meanWords_User'], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.5)
plt.title('Reviewer Clusters (K-means)')
plt.xlabel('Average Reviews Per Day')
plt.ylabel('Average Word Count')
plt.xticks('avgPerDay_User')
plt.yticks('meanWords_User')
#plt.legend(loc='lower right')
#plt.show()
#plt.close()


# *****Boxplot Avg Reviews/Day**********************************************


# Boxplot for average reviews/day by user 
df2.boxplot(column = 'avgPerDay_User').plot()
plt.title('Average # of Reviews per Day by User')
#plt.show()
plt.close()



# Boxplot for average avg words per review by user 
df2.boxplot(column = 'meanWords_User').plot()
plt.title('Average # of Words per Review (by User)')
plt.show()
describe = df2.describe()
print(describe)
