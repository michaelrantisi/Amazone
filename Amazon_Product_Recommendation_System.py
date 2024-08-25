#!/usr/bin/env python
# coding: utf-8

# # **Project: Amazon Product Recommendation System**
# 
# The dataset contains ratings of different electronic products. It does not include information about the products or reviews to avoid bias while building the model.
# 
# --------------
# ## **Context:**
# --------------
# 
# Today, information is growing exponentially with volume, velocity and variety throughout the globe. This has lead to information overload, and too many choices for the consumer of any business. It represents a real dilemma for these consumers and they often turn to denial. Recommender Systems are one of the best tools that help recommending products to consumers while they are browsing online. Providing personalized recommendations which is most relevant for the user is what's most likely to keep them engaged and help business.
# 
# E-commerce websites like Amazon, Walmart, Target and Etsy use different recommendation models to provide personalized suggestions to different users. These companies spend millions of dollars to come up with algorithmic techniques that can provide personalized recommendations to their users.
# 
# Amazon, for example, is well-known for its accurate selection of recommendations in its online site. Amazon's recommendation system is capable of intelligently analyzing and predicting customers' shopping preferences in order to offer them a list of recommended products. Amazon's recommendation algorithm is therefore a key element in using AI to improve the personalization of its website. For example, one of the baseline recommendation models that Amazon uses is item-to-item collaborative filtering, which scales to massive data sets and produces high-quality recommendations in real-time.
# 
# ----------------
# ## **Objective:**
# ----------------
# 
# the task of building a recommendation system to recommend products to customers based on their previous ratings for other products. You have a collection of labeled data of Amazon reviews of products. The goal is to extract meaningful insights from the data and build a recommendation system that helps in recommending products to online consumers.
# 
# -----------------------------
# ## **Dataset:**
# -----------------------------
# 
# The Amazon dataset contains the following attributes:
# 
# - **userId:** Every user identified with a unique id
# - **productId:** Every product identified with a unique id
# - **Rating:** The rating of the corresponding product by the corresponding user
# - **timestamp:** Time of the rating. We **will not use this column** to solve the current problem

# In[57]:


from google.colab import drive
drive.mount('/content/drive')


# In[58]:


get_ipython().system('pip install scikit-surprise')



# ## **Importing the necessary libraries and overview of the dataset**

# In[59]:


import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt


# ### **Loading the data**
# - Import the Dataset
# - Add column names ['user_id', 'prod_id', 'rating', 'timestamp']
# - Drop the column timestamp
# - Copy the data to another DataFrame called **df**

# In[60]:


csv_file_path = "/content/drive/MyDrive/ratings_Electronics.csv"
data = pd.read_csv(csv_file_path, names=['user_id', 'prod_id', 'rating', 'timestamp'])
data.drop(columns='timestamp', inplace=True)
df = data.copy()
print(df.head())
print(df.info())


# **As this dataset is very large and has 7,824,482 observations, it is not computationally possible to build a model using this. Moreover, many users have only rated a few products and also some products are rated by very few users. Hence, we can reduce the dataset by considering certain logical assumptions.**
# 
# Here, we will be taking users who have given at least 50 ratings, and the products that have at least 5 ratings, as when we shop online we prefer to have some number of ratings of a product.

# In[61]:


# Get the column containing the users
users = df.user_id

# Create a dictionary from users to their number of ratings
ratings_count = dict()

for user in users:

    # If we already have the user, just add 1 to their rating count
    if user in ratings_count:
        ratings_count[user] += 1

    # Otherwise, set their rating count to 1
    else:
        ratings_count[user] = 1


# In[62]:


# We want our users to have at least 50 ratings to be considered
RATINGS_CUTOFF = 50

remove_users = []

for user, num_ratings in ratings_count.items():
    if num_ratings < RATINGS_CUTOFF:
        remove_users.append(user)

df = df.loc[ ~ df.user_id.isin(remove_users)]


# In[63]:


# Get the column containing the products
prods = df.prod_id

# Create a dictionary from products to their number of ratings
ratings_count = dict()

for prod in prods:

    # If we already have the product, just add 1 to its rating count
    if prod in ratings_count:
        ratings_count[prod] += 1

    # Otherwise, set their rating count to 1
    else:
        ratings_count[prod] = 1


# In[64]:


# We want our item to have at least 5 ratings to be considered
RATINGS_CUTOFF = 5

remove_users = []

for user, num_ratings in ratings_count.items():
    if num_ratings < RATINGS_CUTOFF:
        remove_users.append(user)

df_final = df.loc[~ df.prod_id.isin(remove_users)]


# In[65]:


# Print a few rows of the imported dataset
df_final.head()


# ## **Exploratory Data Analysis**

# ### **Shape of the data**

# In[66]:


num_rows, num_cols = df_final.shape

# Provide observations
print("Number of rows:", num_rows)
print("Number of columns:", num_cols)


# **Write your observations here: The final reduced dataset 'df_final' contains 65,290 user-product interactions with 3 columns: 'user_id', 'prod_id', and 'rating'. The logical filtering assumptions resulted in a manageable dataset focusing on users with at least 50 ratings and products with at least 5 ratings, making it suitable for building a recommendation system.**

# ### **Data types**

# In[67]:


# Check Data types and provide observations
data_types = df_final.dtypes

print(data_types)


# **Write your observations here:The 'user_id' and 'prod_id' columns are of data type 'object', representing strings or categorical data, while the 'rating' column is of data type 'float64', indicating numeric data with decimal values for representing the ratings. These data types are suitable for building a recommendation system based on collaborative filtering techniques.**

# ### **Checking for missing values**

# In[68]:


# Check for missing values present and provide observations
missing_values = df_final.isnull().sum()

print(missing_values)


# **Write your observations here:The output indicates that there are no missing values present in the 'df_final' DataFrame for any of the columns. Each column, namely 'user_id', 'prod_id', and 'rating', has a count of 0 missing values.**

# ### **Summary Statistics**

# In[69]:


# Summary statistics of 'rating' variable and provide observations
rating_stats = df_final['rating'].describe()

print(rating_stats)


# **Write your observations here:The 'rating' variable has 65,290 valid ratings. The average rating is approximately 4.29 with a standard deviation of about 0.99, indicating moderate variability. Ratings range from 1.0 (minimum) to 5.0 (maximum). Most ratings are high, as indicated by the 25th percentile at 4.0, median at 5.0, and 75th percentile at 5.0. This suggests that users generally rate products positively in the reduced dataset.**

# ### **Checking the rating distribution**

# In[70]:


# Create the bar plot and provide observations
plt.figure(figsize=(8, 5))
df_final['rating'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution')
plt.show()


# from the rating distribution bar plot we can see that 5.0 rating is the highest count with a count of over 35000 while 4.0 is around 19000 etc... we see that the highest ratings have the highest count and vice versa *

# ### **Checking the number of unique users and items in the dataset**

# In[71]:


# Number of total rows in the data and number of unique user id and product id in the data
total_rows = df_final.shape[0]
num_unique_users = df_final['user_id'].nunique()
num_unique_products = df_final['prod_id'].nunique()

print("Number of total rows in the data:", total_rows)
print("Number of unique user IDs in the data:", num_unique_users)
print("Number of unique product IDs in the data:", num_unique_products)


# 
# 
# The 'df_final' dataset contains 65,290 user-product interactions. There are 1,540 unique user IDs and 5,689 unique product IDs in the data. This indicates that 1,540 users have rated 5,689 distinct products, forming the basis for the recommendation system.**

# ### **Users with the most number of ratings**

# In[72]:


# Top 10 users based on the number of ratings
top_10_users = df_final['user_id'].value_counts().head(10)

print(top_10_users)


# The output displays the top 10 users in the 'df_final' dataset based on the number of ratings they have provided. Each line represents a user ID along with the count of ratings they have given These top users have contributed a significant number of ratings, making them key contributors to the dataset.**

# 

# ## **Model 1: Rank Based Recommendation System**

# In[73]:


# Calculate the average rating for each product
average_rating = df_final.groupby('prod_id')['rating'].mean()
# Calculate the count of ratings for each product
count_ratings = df_final.groupby('prod_id')['rating'].count()
# Create a dataframe with calculated average and count of ratings
final_rating = pd.DataFrame({'Average_Rating': average_rating, 'Rating_Count': count_ratings})
# Sort the dataframe by average of ratings in the descending order
final_rating = final_rating.sort_values(by='Average_Rating', ascending=False)
# See the first five records of the "final_rating" dataset
print(final_rating.head())


# In[74]:


# Function to get the top n products based on the highest average rating and minimum interactions
def get_top_n_products(df, n, min_interactions):
    # Calculate the average rating for each product
    average_rating = df.groupby('prod_id')['rating'].mean()

    # Calculate the count of ratings for each product
    count_ratings = df.groupby('prod_id')['rating'].count()

    # Create a DataFrame with calculated average and count of ratings
    final_rating = pd.DataFrame({'Average_Rating': average_rating, 'Rating_Count': count_ratings})

    # Finding products with minimum number of interactions (minimum rating count)
    min_interactions = final_rating['Rating_Count'].min()

    # Filter products with minimum interactions
    filtered_products = final_rating[final_rating['Rating_Count'] == min_interactions]

    # Sort the DataFrame with respect to average rating in descending order
    sorted_products = filtered_products.sort_values(by='Average_Rating', ascending=False)

    # Get the top n products based on the highest average rating and minimum interactions
    top_n_products = sorted_products.head(n)

    return top_n_products



# ### **Recommending top 5 products with 50 minimum interactions based on popularity**

# In[75]:


top_5_popular_products = get_top_n_products(df_final, n=5, min_interactions=50)

print(top_5_popular_products)


# ### **Recommending top 5 products with 100 minimum interactions based on popularity**

# In[76]:


top_5_popular_products = get_top_n_products(df_final, n=5, min_interactions=100)

print(top_5_popular_products)


# ## **Model 2: Collaborative Filtering Recommendation System**

# ### **Building a baseline user-user similarity based recommendation system**

# In[77]:


from surprise import accuracy
from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import KFold
from surprise import CoClustering


# In[79]:


def precision_recall_at_k(model, k = 10, threshold = 3.5):
    """Return precision and recall at k metrics for each user"""

    user_est_true = defaultdict(list)

    predictions = model.test(testset)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        user_ratings.sort(key = lambda x: x[0], reverse = True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    precision = round((sum(prec for prec in precisions.values()) / len(precisions)), 3)

    recall = round((sum(rec for rec in recalls.values()) / len(recalls)), 3)

    accuracy.rmse(predictions)

    print('Precision: ', precision)

    print('Recall: ', recall)

    print('F_1 score: ', round((2*precision*recall)/(precision+recall), 3))


# In[80]:


# Instantiating Reader scale with expected rating scale
reader = Reader(rating_scale=(1, 5))
# Loading the rating dataset
data = Dataset.load_from_df(df_final[['user_id', 'prod_id', 'rating']], reader)
# Splitting the data into train and test datasets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# ### **Building the user-user Similarity-based Recommendation System**

# In[81]:


from surprise.prediction_algorithms.knns import KNNBasic
from collections import defaultdict
sim_options = {'name': 'cosine', 'user_based': True}

model_user_user = KNNBasic(sim_options=sim_options, verbose=False, random_state=1)

model_user_user.fit(trainset)

precision_recall_at_k(model_user_user, k=10, threshold=3.5)


# The user-user similarity-based recommendation system using the KNNBasic model achieved good performance These metrics indicate that the model's predictions are relatively close to the true ratings, and it provides relevant recommendations to users with high precision and recall. Overall, the system shows effectiveness in suggesting products based on user similarities and rating patterns.**

# In[82]:


# Predicting rating for a sample user with an interacted product
def get_predicted_rating(model, user_id, prod_id):
    prediction = model.predict(user_id, prod_id)
    return round(prediction.est, 2)

user_id_sample = 'A3LDPF5FMB782Z'
prod_id_sample = '1400501466'

predicted_rating = get_predicted_rating(model_user_user, user_id_sample, prod_id_sample)

print(f"Predicted rating for user {user_id_sample} and product {prod_id_sample}: {predicted_rating}")


# The user-user similarity-based recommendation system predicted a rating of 3.4 for the product with productId '1400501466' by the user with userId 'A3LDPF5FMB782Z'. This rating estimation is based on the model's understanding of user preferences and similarities with other users.**

# Below is the **list of users who have not seen the product with product id "1400501466"**.

# In[83]:


# Find unique user_id where prod_id is not equal to "1400501466"
unique_users_not_seen_product = df_final[df_final['prod_id'] != "1400501466"]['user_id'].unique()

print(unique_users_not_seen_product)


# In[84]:


# Predicting rating for a sample user with a non interacted product
user_id_sample = 'A34BZM6S9L7QI4'
prod_id_sample = '1400501466'
predicted_rating = get_predicted_rating(model_user_user, user_id_sample, prod_id_sample)

print(f"Predicted rating for user {user_id_sample} and product {prod_id_sample}: {predicted_rating}")


# The user-user similarity-based recommendation system predicted a rating of 4.29 for the product with productId '1400501466' by the user with userId 'A34BZM6S9L7QI4'. This rating estimation is based on the model's understanding of user preferences and similarities with other users, even though the user has not interacted with the specific product.**

# ### **Improving similarity-based recommendation system by tuning its hyperparameters**

# In[85]:


from surprise.model_selection import GridSearchCV
from surprise.dataset import Dataset
from surprise.reader import Reader
from surprise import KNNBasic

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_final[['user_id', 'prod_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

param_grid = {
    'k': [10, 20, 30, 40],
    'min_k': [1, 2, 3],
    'sim_options': {
        'name': ['cosine', 'msd', 'pearson', 'pearson_baseline'],
        'user_based': [True],
    }
}
grid_search = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3)

grid_search.fit(data)

best_rmse = grid_search.best_score['rmse']

best_params = grid_search.best_params['rmse']

print(f"Best RMSE score: {best_rmse}")
print(f"Best combination of hyperparameters: {best_params}")



# In[86]:


from surprise.prediction_algorithms.knns import KNNBasic
similarity_measure = 'cosine'
k_neighbors = 10
min_k_neighbors = 1

model_final = KNNBasic(k=k_neighbors, min_k=min_k_neighbors, sim_options={'name': similarity_measure, 'user_based': True})

model_final.fit(trainset)

def get_top_k_recommendations(model, user_id, k=10):
    test_user = trainset.to_inner_uid(user_id)
    predictions = model.test(testset)
    top_k_predictions = [pred for pred in predictions if pred.uid == test_user]
    top_k_predictions.sort(key=lambda x: x.est, reverse=True)
    top_k_product_ids = [trainset.to_raw_iid(pred.iid) for pred in top_k_predictions[:k]]
    return top_k_product_ids
user_id_sample = 'A34BZM6S9L7QI4'
top_k_recommendations = get_top_k_recommendations(model_final, user_id_sample, k=10)




# The optimized user-user similarity-based recommendation system achieved a best RMSE score of approximately 0.9714. The hyperparameters that resulted in this score are k=40, min_k=3, and using the cosine similarity measure for user-based collaborative filtering. These settings allow the model to make more accurate predictions and generate personalized product recommendations for users.**

# In[87]:


# Use sim_user_user_optimized model to recommend for userId "A3LDPF5FMB782Z" and productId 1400501466
test_user_id = 'A3LDPF5FMB782Z'
test_prod_id = '1400501466'
test_user_inner_id = model_final.trainset.to_inner_uid(test_user_id)
test_prod_inner_id = model_final.trainset.to_inner_iid(test_prod_id)

predicted_rating_optimized = model_final.predict(test_user_inner_id, test_prod_inner_id).est

print(f"Predicted rating for user {test_user_id} and product {test_prod_id} using the optimized model: {predicted_rating_optimized:.2f}")


# In[88]:


# Predict rating for user "A34BZM6S9L7QI4" and prod_id="1400501466" using the optimized model
test_user_id = 'A34BZM6S9L7QI4'
test_prod_id = '1400501466'

test_user_inner_id = model_final.trainset.to_inner_uid(test_user_id)
test_prod_inner_id = model_final.trainset.to_inner_iid(test_prod_id)

if not model_final.trainset.knows_user(test_user_inner_id) or not model_final.trainset.knows_item(test_prod_inner_id):
    predicted_rating_optimized = model_final.predict(test_user_inner_id, test_prod_inner_id).est
else:
    predicted_rating_optimized = None

print(f"Predicted rating for user {test_user_id} and product {test_prod_id} using the optimized model: {predicted_rating_optimized}")


# The optimized model predicts that user "A3LDPF5FMB782Z" is likely to rate the product "1400501466" with a rating of approximately 4.29, indicating a positive recommendation for the product.

# ### **Identifying similar users to a given user (nearest neighbors)**

# In[89]:


from surprise.prediction_algorithms.knns import KNNBasic
user_id = 0

try:
    user_inner_id = model_final.trainset.to_inner_uid(user_id)
except ValueError:
    print("User", user_id, "is not part of the trainset.")
else:

    neighbors = model_final.get_neighbors(user_inner_id, k=5)

    similar_users = [model_final.trainset.to_raw_uid(inner_id) for inner_id in neighbors]

    print("5 most similar users to user 0:")
    print(similar_users)


# ### **Implementing the recommendation algorithm based on optimized KNNBasic model**

# In[91]:


def get_recommendations(data, user_id, top_n, algo):

    recommendations = []

    user_item_interactions_matrix = data.pivot(index = 'user_id', columns = 'prod_id', values = 'rating')

    non_interacted_products = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()

    for item_id in non_interacted_products:

        est = algo.predict(user_id, item_id).est

    recommendations.append((item_id, est))

    recommendations.sort(key = lambda x: x[1], reverse = True)

    return recommendations[:top_n]


# **Predicting top 5 products for userId = "A3LDPF5FMB782Z" with similarity based recommendation system**

# In[92]:


from surprise.prediction_algorithms.knns import KNNBasic
algo = KNNBasic(k=40, min_k=3, sim_options={'name': 'cosine', 'user_based': True})
trainset = data.build_full_trainset()
algo.fit(trainset)

data_df = pd.DataFrame(data.raw_ratings, columns=["user_id", "prod_id", "rating", "timestamp"])

data_df.drop(columns="timestamp", inplace=True)

user_id = "A3LDPF5FMB782Z"
top_n = 5
recommendations = get_recommendations(data_df, user_id, top_n, algo)

print("Top 5 recommended products for user A3LDPF5FMB782Z:")
for item_id, est in recommendations:
    print(f"Product ID: {item_id}, Predicted Rating: {est}")




# In[93]:


# Building the dataframe for above recommendations with columns "prod_id" and "predicted_ratings"
import pandas as pd

def get_recommendations(data, user_id, top_n, algo):

    recommendations = []

    user_item_interactions_matrix = data.pivot(index='user_id', columns='prod_id', values='rating')
    non_interacted_products = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()

    for item_id in non_interacted_products:

        est = algo.predict(user_id, item_id).est

        recommendations.append((item_id, est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    df_recommendations = pd.DataFrame(recommendations, columns=["prod_id", "predicted_ratings"])

    return df_recommendations[:top_n]


# ### **Item-Item Similarity-based Collaborative Filtering Recommendation System**

# In[94]:


from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import train_test_split
import pandas as pd

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=1)

sim_options = {
    'name': 'cosine',
    'user_based': False
}

algo_item_item = KNNBasic(sim_options=sim_options, random_state=1)

algo_item_item.fit(trainset)
predictions_item_item = algo_item_item.test(testset)

k = 10

precision = 0
recall = 0
for uid, _, true_r, est, _ in predictions_item_item:
    if est >= 3.5:
        precision += (true_r >= 3.5)
        recall += 1

precision_at_k = precision / len(predictions_item_item)
recall_at_k = recall / len(predictions_item_item)

f1_score_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

print(f"Precision@{k}: {precision_at_k:.3f}")
print(f"Recall@{k}: {recall_at_k:.3f}")
print(f"F1-score@{k}: {f1_score_at_k:.3f}")





# The item-item similarity-based collaborative filtering recommendation system performed well with a Precision@10 of 0.797, Recall@10 of 0.947, and F1-score@10 of 0.865. This indicates that the system provided relevant and accurate recommendations to the user. Further fine-tuning may enhance its performance.

# In[95]:


from surprise import Dataset, Reader, KNNBasic

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

trainset = data.build_full_trainset()
algo = KNNBasic(k=40, min_k=3, sim_options={'name': 'cosine', 'user_based': True})
algo.fit(trainset)

user_id = "A3LDPF5FMB782Z"
prod_id = "1400501466"
predicted_rating = algo.predict(user_id, prod_id).est

print(f"Predicted rating for user {user_id} and product {prod_id}: {predicted_rating:.2f}")



# The predicted rating for user "A3LDPF5FMB782Z" and product "1400501466" is approximately 3.33. This suggests that the user's preference for the product is slightly above average. However, the prediction may not perfectly reflect the user's actual opinion, and further evaluation is needed to assess the algorithm's performance accurately.
# 
# 
# 

# In[96]:


from surprise import Dataset, Reader, KNNBasic

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

trainset = data.build_full_trainset()
algo = KNNBasic(k=40, min_k=3, sim_options={'name': 'cosine', 'user_based': True})
algo.fit(trainset)

user_id = "A34BZM6S9L7QI4"
prod_id = "1400501466"
predicted_rating = algo.predict(user_id, prod_id).est

print(f"Predicted rating for user {user_id} and product {prod_id}: {predicted_rating:.2f}")




# The predicted rating for the user "A34BZM6S9L7QI4" and the product "1400501466" is 4.29. The high predicted rating suggests that the user is likely to have a positive experience with the product.

# ### **Hyperparameter tuning the item-item similarity-based model**
# 

# In[97]:


from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import GridSearchCV

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

param_grid = {
    'k': [10, 20, 30],
    'min_k': [3, 6, 9],
    'sim_options': {'name': ['msd', 'cosine'], 'user_based': [False]}
}

algo = KNNBasic()

grid_search = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3)
grid_search.fit(data)

best_rmse = grid_search.best_score['rmse']
best_params = grid_search.best_params['rmse']

print(f"Best RMSE score: {best_rmse:.3f}")
print("Best parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")


# ### ** the best parameters from GridSearchCV to build the optimized item-item similarity-based model and Comparing the performance of the optimized model with the baseline model.**

# In[98]:


from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=1)

best_params = grid_search.best_params['rmse']
sim_options = {
    'name': best_params['sim_options']['name'],
    'user_based': False
}
algo_optimized = KNNBasic(k=best_params['k'], min_k=best_params['min_k'], sim_options=sim_options)

algo_optimized.fit(trainset)

predictions_optimized = algo_optimized.test(testset)

k = 10
relevant_predictions = [pred for pred in predictions_optimized if pred.r_ui >= 3.5]
num_relevant_items = len(relevant_predictions)
num_total_items = len(predictions_optimized)
precision_at_k_optimized = num_relevant_items / k
recall_at_k_optimized = num_relevant_items / num_total_items

f1_score_optimized = 2 * (precision_at_k_optimized * recall_at_k_optimized) / (precision_at_k_optimized + recall_at_k_optimized)
rmse_optimized = accuracy.rmse(predictions_optimized)

print("Optimized model:")
print(f"Precision@{k}: {precision_at_k_optimized:.3f}")
print(f"Recall@{k}: {recall_at_k_optimized:.3f}")
print(f"F1-score@{k}: {f1_score_optimized:.3f}")
print(f"RMSE: {rmse_optimized:.3f}")



# The optimized item-item similarity-based model performs better than the baseline model in terms of precision, recall, and F1-score. It also shows good accuracy in predicting ratings with a lower RMSE value.**

# In[99]:


# Use sim_item_item_optimized model to recommend for userId "A3LDPF5FMB782Z" and productId "1400501466"
user_id = "A3LDPF5FMB782Z"
prod_id = "1400501466"
user_item_df = pd.DataFrame({'user_id': [user_id], 'prod_id': [prod_id]})

predicted_rating_optimized = algo_optimized.predict(user_id, prod_id).est

predicted_rating_baseline = algo_item_item.predict(user_id, prod_id).est

print(f"Predicted rating for user {user_id} and product {prod_id} using the optimized model: {predicted_rating_optimized:.2f}")
print(f"Predicted rating for user {user_id} and product {prod_id} using the baseline model: {predicted_rating_baseline:.2f}")


# In[100]:


# Use sim_item_item_optimized model to recommend for userId "A34BZM6S9L7QI4" and productId "1400501466"
user_id = "A34BZM6S9L7QI4"
prod_id = "1400501466"
user_item_df = pd.DataFrame({'user_id': [user_id], 'prod_id': [prod_id]})

predicted_rating_optimized = algo_optimized.predict(user_id, prod_id).est
predicted_rating_baseline = algo_item_item.predict(user_id, prod_id).est

print(f"Predicted rating for user {user_id} and product {prod_id} using the optimized model: {predicted_rating_optimized:.2f}")
print(f"Predicted rating for user {user_id} and product {prod_id} using the baseline model: {predicted_rating_baseline:.2f}")


# The optimized model generally provides higher predicted ratings compared to the baseline model for user A3LDPF5FMB782Z and product 1400501466. However, for user A34BZM6S9L7QI4 and product 1400501466, both models predict similar ratings, with the optimized model having a slight advantage with a rating of 4.30 compared to the baseline model's rating of 4.00.**

# ### **Identifying similar items to a given item (nearest neighbors)**

# In[101]:


from surprise import Dataset, Reader, KNNBasic

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

trainset = data.build_full_trainset()
algo = KNNBasic(k=40, min_k=3, sim_options={'name': 'msd', 'user_based': False})
algo.fit(trainset)
item_raw_id = trainset.to_raw_iid(0)

item_neighbors = algo.get_neighbors(trainset.to_inner_iid(item_raw_id), k=5)

similar_items = [trainset.to_raw_iid(inner_id) for inner_id in item_neighbors]

print(f"The 5 most similar items to item with id {item_raw_id} are: {similar_items}")


# **Predicting top 5 products for userId = "A1A5KUIIIHFF4U" with similarity based recommendation system.**

# In[102]:


from surprise import Dataset, Reader, KNNBasic

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

trainset = data.build_full_trainset()
algo = KNNBasic(k=40, min_k=3, sim_options={'name': 'cosine', 'user_based': True})
algo.fit(trainset)

def get_recommendations(data, user_id, top_n, algo):

    recommendations = []

    all_product_ids = data_df['prod_id'].unique()

    for prod_id in all_product_ids:
        est = algo.predict(user_id, prod_id).est

        recommendations.append((prod_id, est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [prod_id for prod_id, _ in recommendations[:top_n]]

user_id = "A1A5KUIIIHFF4U"
top_n = 5
recommendations = get_recommendations(data_df, user_id, top_n, algo)

print(f"Top 5 recommended products for user {user_id}: {recommendations}")


# In[104]:


import pandas as pd
from surprise import Dataset, Reader, KNNBasic

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

trainset = data.build_full_trainset()
algo = KNNBasic(k=40, min_k=3, sim_options={'name': 'cosine', 'user_based': True})
algo.fit(trainset)

def get_recommendations(data, user_id, top_n, algo):
    recommendations = []

    all_product_ids = data_df['prod_id'].unique()

    for prod_id in all_product_ids:
        est = algo.predict(user_id, prod_id).est
        recommendations.append((prod_id, est))
    recommendations.sort(key=lambda x: x[1], reverse=True)

    df_recommendations = pd.DataFrame(recommendations, columns=["prod_id", "predicted_ratings"])

    return df_recommendations.head(top_n)

user_id = "A1A5KUIIIHFF4U"
top_n = 5
recommendations_df = get_recommendations(data_df, user_id, top_n, algo)

print(f"Top 5 recommended products for user {user_id}:\n{recommendations_df}")


# ### **Model 3: Model-Based Collaborative Filtering - Matrix Factorization**

# ### Singular Value Decomposition (SVD)

# In[105]:


from surprise import SVD
from surprise import accuracy
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=1)

algo_svd = SVD(random_state=1)

algo_svd.fit(trainset)

predictions_svd = algo_svd.test(testset)

y_true = [pred.r_ui for pred in predictions_svd]
y_pred = [pred.est for pred in predictions_svd]

threshold = 3.5
binary_y_true = [1 if true_rating >= threshold else 0 for true_rating in y_true]
binary_y_pred = [1 if pred_rating >= threshold else 0 for pred_rating in y_pred]

precision_svd = precision_score(binary_y_true, binary_y_pred)
recall_svd = recall_score(binary_y_true, binary_y_pred)
f1_score_svd = f1_score(binary_y_true, binary_y_pred)

rmse_svd = accuracy.rmse(predictions_svd)

print("SVD Model Metrics:")
print(f"Precision: {precision_svd:.3f}")
print(f"Recall: {recall_svd:.3f}")
print(f"F1-score: {f1_score_svd:.3f}")
print(f"RMSE: {rmse_svd:.3f}")



# The SVD model-based collaborative filtering approach performs well with an RMSE of 0.9104. The precision is 0.847, indicating around 84.7% of the recommended items are relevant, while the recall is 0.974, suggesting successful recommendations of about 97.4% of the relevant items. The F1-score is 0.907, striking a good balance between precision and recall. Overall, the SVD model provides accurate and personalized recommendations based on latent features in the user-item matrix.**

# In[106]:


predicted_rating_svd = algo_svd.predict("A3LDPF5FMB782Z", "1400501466").est

print(f"Predicted rating for user A3LDPF5FMB782Z and product 1400501466: {predicted_rating_svd:.2f}")



# The SVD model predicts a rating of 4.09 for user "A3LDPF5FMB782Z" and product "1400501466". The predicted rating suggests a positive opinion from the user for the product.**

# In[107]:


user_id = "A34BZM6S9L7QI4"
prod_id = "1400501466"

predicted_rating = algo_svd.predict(user_id, prod_id).est

print(f"Predicted rating for user {user_id} and product {prod_id}: {predicted_rating:.2f}")


# The predicted rating for user "A34BZM6S9L7QI4" and product "1400501466" using the SVD model is 4.05. This predicted rating suggests that the model believes user "A34BZM6S9L7QI4" is likely to rate the product "1400501466" with a rating of 4.05 on a scale of 1 to 5. It indicates that the model considers this user-product interaction to be positive and recommends this product to the user with a relatively high rating.**

# ### **Improving Matrix Factorization based recommendation system by tuning its hyperparameters**

# In[108]:


from surprise.model_selection import GridSearchCV
from surprise import SVD

param_grid = {
    'n_epochs': [5, 10, 15],
    'lr_all': [0.001, 0.005, 0.01],
    'reg_all': [0.01, 0.02, 0.05]
}


algo_svd = SVD()

grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid_search.fit(data)

best_rmse_score = grid_search.best_score['rmse']
best_params = grid_search.best_params['rmse']

print(f"Best RMSE score: {best_rmse_score:.4f}")
print("Best parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")



# In[109]:


from surprise import SVD
from surprise.model_selection import GridSearchCV
from surprise import Dataset, Reader, accuracy
from collections import defaultdict

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data_df[['user_id', 'prod_id', 'rating']], reader)

param_grid = {
    'n_epochs': [5, 10, 15],
    'lr_all': [0.001, 0.005, 0.01],
    'reg_all': [0.01, 0.02, 0.05]
}

grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid_search.fit(data)
best_rmse_score = grid_search.best_score['rmse']
best_params = grid_search.best_params['rmse']

print(f"Best RMSE score: {best_rmse_score:.4f}")
print("Best parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
svd_optimized = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'], random_state=1)

trainset = data.build_full_trainset()
svd_optimized.fit(trainset)

testset = trainset.build_anti_testset()
predictions_svd_optimized = svd_optimized.test(testset)

k = 10

precision = 0
recall = 0
for uid, _, true_r, est, _ in predictions_item_item:
    if est >= 3.5:
        precision += (true_r >= 3.5)
        recall += 1

precision_at_k = precision / len(predictions_item_item)
recall_at_k = recall / len(predictions_item_item)

f1_score_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

print(f"Precision@{k}: {precision_at_k:.3f}")
print(f"Recall@{k}: {recall_at_k:.3f}")
print(f"F1-score@{k}: {f1_score_at_k:.3f}")





# 
# The optimized SVD model achieved a good RMSE score of 0.9050. For the top 10 recommendations, it had a precision of 0.797 and a recall of 0.947, resulting in an overall F1-score of 0.865. This indicates the model's effectiveness in providing accurate and relevant recommendations based on user preferences and item ratings.**
# 
# 
# 

# In[110]:


svd_optimized = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'], random_state=1)

trainset = data.build_full_trainset()
svd_optimized.fit(trainset)

testset = trainset.build_anti_testset()

user_id = "A3LDPF5FMB782Z"
prod_id = "1400501466"

estimated_rating_svd_optimized = svd_optimized.predict(user_id, prod_id).est

print(f"Estimated rating using the optimized SVD model: {estimated_rating_svd_optimized:.2f}")


# In[111]:


# Use svd_algo_optimized model to recommend for userId "A34BZM6S9L7QI4" and productId "1400501466"
user_id = "A34BZM6S9L7QI4"
prod_id = "1400501466"

estimated_rating_svd_optimized = svd_optimized.predict(user_id, prod_id).est

print(f"Estimated rating using the optimized SVD model: {estimated_rating_svd_optimized:.2f}")


# ### **Conclusion and Recommendations**

# 
# in this task of building a recommendation system for Amazon based on customer ratings, we used the Surprise library to implement collaborative filtering using the Singular Value Decomposition (SVD) algorithm. We started by creating a rating dataset from the Amazon reviews data using the Surprise Reader and Dataset classes. Then, we performed hyperparameter tuning using GridSearchCV to find the best parameters for the SVD model based on the root mean squared error (RMSE) metric.
# 
# After tuning the hyperparameters, we built an optimized SVD model using the best parameters and trained it on the full dataset. We then made predictions on the test set and evaluated the model's performance using precision@k and recall@k metrics to measure the top-k recommendation quality. Additionally, we computed the F1-score to balance precision and recall.
# 
# Recommendations:
# 
# Deploy Optimized SVD Model: Implement the optimized SVD model on the Amazon website to provide personalized product recommendations to customers.
# 
# Monitor User Interactions: Regularly monitor user interactions and feedback to gather data and refine the recommendation system.
# 
# Incorporate Additional Data: Explore using additional user-related data, such as demographics and browsing behavior, to enhance the model's understanding of individual preferences.
# 
# Real-Time Updates: Utilize real-time data to continually update the recommendation system and improve accuracy.
# 
# A/B Testing: Conduct A/B testing to assess the impact of the recommendation system on user engagement and conversion rates.
# 
# Explainability: Consider adding an explanation module to provide users with insights into why specific products are recommended.
# 
# Privacy and Security: Ensure customer data privacy and security in accordance with best practices and regulations.

# In[ ]:




