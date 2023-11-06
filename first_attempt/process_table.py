import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

USER_IN_ROOMS = 'user_in_room_last_30_days_fullsize.csv'
ROOM_FEATURES = 'room_features_in_last_30_days.json'
user_in_rooms = pd.read_csv(USER_IN_ROOMS)
room_features = pd.read_json(ROOM_FEATURES)
room_usable_features = ['room_id','championship_group' ,'team_ids', 'room_seasonality', 'player_ids' ,'category', 'type', 'buyin', 'is_beginner_room', 'guaranteed_prize', 'room_capacity', 'max_user_lineups']



# Only keeping useful rows championship_group, room_seasonality, category, type, buyin, is_beginner_room, guaranteed_prize, room_capacity, max_user_lineups:
def remove_unnecessary_columns(room_features):
    for column in room_features.columns:
        if column not in room_usable_features:
            room_features.drop(column, axis=1, inplace=True)
    return room_features


def categorize_column(column, input_df, is_array = False):
    categorized_df = input_df.copy()
    for index, row in input_df.iterrows():
        if is_array:
            for item in row[column]:
                categorized_df.at[index, column + '_' + str(item)] = 1
        else:
            categorized_df.at[index, column + '_' + str(row[column])] = 1
    categorized_df = categorized_df.fillna(0)
    categorized_df.drop(column, axis=1, inplace=True)
    return categorized_df

# create a df where we get for each user, a column with the sum of the features of the rooms he played in
def create_df_with_sum_of_features(user_in_rooms, room_features):
    print('over merged_data')
    # Merge the user_in_room DataFrame with room_features based on Room_ID
    merged_data = pd.merge(user_in_rooms, room_features, on='room_id')
    print('under merged_data')
    # Group by User_ID and sum the features for each user
    user_features = merged_data.groupby('user_id').sum().reset_index()
    print('under mgroup user features')

    # Reset the index to make User_ID a regular column
    user_features.reset_index(drop=True, inplace=True)
    user_features.drop('room_id', axis=1, inplace=True)
    user_features.drop('user_id', axis=1, inplace=True)

    return user_features




def process_data(user_in_rooms, room_features):
    print('processing data')
    # Removing rows with empty matches ( rooms are not valid )
    room_features = room_features[room_features['match_id_formatted'].apply(lambda x: len(x) > 0)]

    room_features = remove_unnecessary_columns(room_features)

    # room_usable_features = ['room_id','championship_group','team_ids' ,'room_seasonality', 'category', 'type', 'buyin', 'is_beginner_room', 'guaranteed_prize', 'room_capacity', 'max_user_lineups']

    room_features = categorize_column('team_ids', room_features, True)

    print('hi')
    room_features = categorize_column('player_ids', room_features, True)
    print('hi')

    room_features = categorize_column('category', room_features)

    room_features = categorize_column('championship_group', room_features)

    room_features = categorize_column('buyin', room_features)

    room_features = categorize_column('room_seasonality', room_features)

    room_features = categorize_column('type', room_features)

    room_features = categorize_column('max_user_lineups', room_features)


    room_features = categorize_column('guaranteed_prize', room_features)

    room_features = categorize_column('room_capacity', room_features)

    # turn boolean column into int column
    room_features['is_beginner_room'] = room_features['is_beginner_room'].astype(int)

    room_features.fillna(0, inplace=True)
    user_in_rooms.drop('lineup_id', axis=1, inplace=True)
    user_in_rooms.drop('lineup_create_date', axis=1, inplace=True)
    print('hi')

    # print room features to csv
    # room_features.to_csv('room_features.csv', index=False)

    user_by_features = create_df_with_sum_of_features(user_in_rooms, room_features)

    # user_by_features.to_csv('user_by_features.csv', index=False)
    # Make an array with the sum of the columns for each feature
    column_sum = user_by_features.sum(axis=0)
    print('hi')
    #sort the array
    column_sum.sort_values(ascending=False, inplace=True)
    #print into a file in a way you can see the feature name
    # Create a new DataFrame with column names and their corresponding sums
    column_sum = pd.DataFrame({'Column Name': column_sum.index, 'Sum': column_sum.values})

    # Save the new DataFrame to a CSV file
    print('hi')
    column_sum.to_csv('column_sums_v2.csv', index=False)




def print_unique_size(column, df):
    print(df[column].unique())



if __name__ == '__main__':
    print(room_features)
    # print_unique_size('guaranteed_prize', room_features)

    user_by_features = process_data(user_in_rooms, room_features)
