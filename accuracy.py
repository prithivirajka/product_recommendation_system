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

#spliting dataset into train and test for evaluation
from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings, test_size = 0.2, random_state = 0)

#resetting the index of train and test
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

#grouping all users based and calculating the mean rating of each user
rating_mean_split = train.groupby(by="user_id",as_index=False)['rating'].mean()

#subtracting each users rating for different items by mean rating of each users
average_rating_split = pd.merge(train,rating_mean_split,on='user_id')
average_rating_split['average_rating']=average_rating_split['rating_x']-average_rating_split['rating_y']

#creating a pivot table with user_id as index, movie_id as columns and average ratings as values
rating_pivot_split = average_rating_split.pivot(index='user_id', columns='movie_id', values='average_rating')

#'Nan' values are replaced with mean values of each rows
final_split = rating_pivot_split.apply(lambda row: row.fillna(row.mean()), axis=1)

#cosine similarity is calculated for each users to users 
similarity_split = 1 -cosine_similarity(final_split)
np.fill_diagonal(similarity_split, 0 )
similarity_with_user_split = pd.DataFrame(similarity_split,index=final_split.index)
similarity_with_user_split.columns=final_split.index

#Function to get the rating given by a user to a movie.
def get_rating_split(userid,movieid):
    return (train.loc[(train.user_id==userid) & (train.movie_id == movieid),'rating'].iloc[0])

#Function to get the average rating of a user.
def get_average_rating_split(userid):
    return (rating_mean_split.loc[(rating_mean_split.user_id==userid),'rating'].iloc[0])

#function to get top 5 similar users
def most_similar_users_split(userid):
    user_ids = train.user_id.unique().tolist()
    similarity_score = [((similarity_with_user_split[userid][nth_user]),nth_user) for nth_user in user_ids[:100] if nth_user != userid]
    similarity_score.sort()
    similarity_score.reverse()
    return similarity_score[:5]


#calculating rating given the userid and itemid
def calculating_rating_split(userid, itemid):
    user_and_similarity = most_similar_users_split(userid)
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
            rate = get_rating_split(userid, itemid)
            userrating.append(rate)
        except:
            avg_rate = get_average_rating_split(userid)
            userrating.append(avg_rate)
    dot_product = np.dot(similarity_extract, userrating)
    sum_of_sim = sum(similarity_extract)
    average = dot_product/sum_of_sim
    return int(round(average))

#getting the predicted rating of test in a array
def prediction_test_array(a):
    prediction_array = []
    print("Prediction test array")
    for i in tqdm(range(a)):
        user = test.loc[i].iloc[0]
        item = test.loc[i].iloc[1]
        prediction = calculating_rating_split(user,item)
        prediction_array.append(prediction)
    predicted_test_values = np.array(prediction_array)
    return predicted_test_values

#getting the given rating of test in a array
def given_test_array(a):
    given_array = []
    for i in range(a):
        rating = test.loc[i].iloc[2]
        given_array.append(rating)
    given_test_values = np.array(given_array)
    return given_test_values

given_rating = given_test_array(len(test))
predicted_rating = prediction_test_array(len(test))

# calculate accuracy
from sklearn import metrics
print("Accuracy for split test:",metrics.accuracy_score(given_rating, predicted_rating))