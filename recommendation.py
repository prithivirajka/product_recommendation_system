#making necesarry imports
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

#importing given dataset and changing the column names to user_id, movie_id, rating
dataset=pd.read_csv("train.txt", header=None, delimiter=" ")
ratings = pd.DataFrame(dataset)
column_indices = [0, 1, 2]
new_names = ['user_id', 'movie_id', 'rating']
old_names = ratings.columns[column_indices]
ratings.rename(columns=dict(zip(old_names, new_names)), inplace=True)

#grouping all users based and calculating the mean rating of each user
rating_mean = ratings.groupby(by="user_id",as_index=False)['rating'].mean()

#subtracting each users rating for different items by mean rating of each users
average_rating = pd.merge(ratings,rating_mean,on='user_id')
average_rating['average_rating']=average_rating['rating_x']-average_rating['rating_y']

#creating a pivot table with user_id as index, movie_id as columns and average ratings as values
rating_pivot = average_rating.pivot(index='user_id', columns='movie_id', values='average_rating')

#'Nan' values are replaced with mean values of each rows
final_pivot = rating_pivot.apply(lambda row: row.fillna(row.mean()), axis=1)

#cosine similarity is calculated for each users to users 
user_similarity = 1 -cosine_similarity(final_pivot)
np.fill_diagonal(user_similarity, 0)
similarity_with_user = pd.DataFrame(user_similarity, index=final_pivot.index)
similarity_with_user.columns=final_pivot.index

#Function to get the rating given by a user to a movie.
def get_rating(userid,movieid):
    return (ratings.loc[(ratings.user_id==userid) & (ratings.movie_id == movieid),'rating'].iloc[0])

#Function to get the average rating of a user.
def get_average_rating(userid):
    return (rating_mean.loc[(rating_mean.user_id==userid),'rating'].iloc[0])

#function to get top 5 similar users
def similar_users(userid):
    user_ids = ratings.user_id.unique().tolist()
    similarity_score = [((similarity_with_user[userid][i]),i) for i in user_ids[:100] if i != userid]
    similarity_score.sort()
    similarity_score.reverse()
    return similarity_score[:5]

#calculating rating given the userid and itemid
def calculating_rating(userid, itemid):
    user_and_similarity = similar_users(userid)
    userextract = []
    for i in range(len(user_and_similarity)):
        u = user_and_similarity[i][1]
        userextract.append(u)
    similarity_extract = []
    for i in range(len(user_and_similarity)):
        u = user_and_similarity[i][0]
        similarity_extract.append(u)
    userrating = []
    for userid in userextract:
        try:
            rate = get_rating(userid, itemid)
            userrating.append(rate)
        except:
            avg_rate = get_average_rating(userid)
            userrating.append(avg_rate)
    dot_product = np.dot(similarity_extract, userrating)
    sum_of_sim = sum(similarity_extract)
    average = dot_product/sum_of_sim
    return int(round(average))

#creating full dataset wil 943 users and 1682 items with zeros in place where no rating is given
def full_df_creation():
    full_df = []
    print("Creating full df with both ratings and zero")
    for i in tqdm(range(943)):
        for j in range(1682):
            try:
                rate = get_rating(i+1, j+1)
                array = [i+1, j+1, rate]
                full_df.append(array)
            except:
                array = [i+1, j+1, 0]
                full_df.append(array)
    return full_df
array = full_df_creation()
df_full = pd.DataFrame(list(array), columns=['user_id', 'movie_id', 'rating'])

#calculating ratings for user and item combination which is not rated 
def getting_fullfinal_dataframe():
    print("predicting rating and replacing with zero")
    for i in tqdm(range(len(df_full))):
        user = df_full.loc[i].iloc[0]
        item = df_full.loc[i].iloc[1]
        rate = df_full.loc[i].iloc[2]
        if rate == 0:
            df_full.loc[i].iloc[2] = calculating_rating(user, item)
        else:
            pass

getting_fullfinal_dataframe()

#copying the dataframe df_full to output.txt
df_full.to_csv('output.txt', header=False, index=False, sep=' ')
