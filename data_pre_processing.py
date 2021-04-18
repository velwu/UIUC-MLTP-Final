import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing

#Import the dataset
def main_proc():
    vgsales = pd.read_csv('vgsales.csv')
    video_game_sales = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
    game_reviews = pd.read_csv('metacritic_critic_reviews.csv')

    #Looks like Vel changed the format of some fields because of some issues, so I keep this for our benefit
    video_game_sales = video_game_sales.replace('tbd', np.nan)
    video_game_sales["User_Score"] = video_game_sales.User_Score.astype(float)
    vgsales['Name'] = vgsales['Name'].str.lower()
    unique_vgsales = vgsales.groupby(['Name','Platform']).count()
    unique_vgsales.reset_index(drop = False, inplace = True)
    unique_vgsales = unique_vgsales.iloc[:,0:2]
    print(unique_vgsales.shape)
    video_game_sales['Name'] = video_game_sales['Name'].str.lower()
    unique_video_game_sales = video_game_sales.groupby(['Name','Platform']).count()
    unique_video_game_sales.reset_index(drop = False, inplace = True)
    unique_video_game_sales = unique_video_game_sales.iloc[:,0:2]
    print(unique_video_game_sales.shape)
    unique_video_game_sales.head()
    #Change the platform format in game_reviews to make them aligned with the other two tables
    game_reviews.replace("PlayStation 4", "PS4", inplace=True)
    game_reviews.replace("PlayStation Vita", "PSV", inplace=True)
    game_reviews.replace("Wii U", "WiiU", inplace=True)
    game_reviews.replace("Xbox One", "XOne", inplace=True)
    #Get the unique game names from game_reviews
    game_reviews['game'] = game_reviews['game'].str.lower()
    unique_game_reviews = game_reviews.groupby(['game','platform']).count()
    unique_game_reviews.reset_index(drop = False, inplace = True)
    unique_game_reviews = unique_game_reviews.iloc[:,0:2]
    print(unique_game_reviews.shape)
    unique_game_reviews.head()

    test_merge3 = pd.merge(game_reviews, video_game_sales, left_on=['game', 'platform'], right_on=['Name','Platform'], how='inner')
    # ideal_df = test_merge3.drop(columns=['Name', 'Platform'])
    ideal_df = test_merge3

    return [unique_vgsales, unique_video_game_sales, unique_game_reviews, video_game_sales, ideal_df]

def filter_by_review_count(input_df, review_count):

    items = input_df.name.value_counts().to_dict().items()
    df_sub = input_df[input_df.name.isin([key for key, val in items if val > review_count])]
    
    return df_sub