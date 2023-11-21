import pandas as pd
from sklearn.metrics import pairwise_distances



USER_IN_ROOMS = 'user_in_room_last_30_days_with_info.csv'
USER_PROFILES_FILEPATH = 'user_profiles_based_on_last_30_days.csv'
NORMALIZATION_PARAMETERS_FILEPATH = 'user_profiles_normalization_parameters.csv'
ROOMS_WITH_INFO_FILEPATH = 'rooms_in_last_30_days_with_info.csv'
SELECTED_ROOMS_WITH_INFO_FILEPATH = 'selected_rooms_with_info.csv'
USER_ROOM_RECOMMENDATIONS_FILEPATH = 'user_room_recommendations.csv'
USER_ROOM_RECOMMENDATIONS_SELECTED_FILEPATH = 'user_room_recommendations_selected.csv'
SELECTED_USERS = 'selected_users.csv'

users_rooms_with_info = pd.read_csv(USER_IN_ROOMS)
user_id_with_room_ids = users_rooms_with_info[['user_id', 'room_id']]
users_rooms_with_info.drop(columns=['room_id'])
target_rooms_with_info = pd.read_csv(SELECTED_ROOMS_WITH_INFO_FILEPATH)
selected_users = pd.read_csv(SELECTED_USERS)

def build_type_features(df):
    features_df = pd.DataFrame()
    features_df['type_UNLIMITED'] = (df['type'] == 'UNLIMITED').astype(int)
    features_df['type_LIMITED'] = (df['type'] == 'LIMITED').astype(int)
    return features_df

def build_capacity_features(df):
    features_df = pd.DataFrame()
    features_df['room_capacity_100'] = (df['room_capacity'] == 100).astype(int)
    return features_df

def build_seasonality_features(df):
    features_df = pd.DataFrame()
    features_df['room_seasonality_DAILY'] = (df['room_seasonality'] == 'DAILY').astype(int)
    features_df['room_seasonality_SINGLE_MATCH'] = (df['room_seasonality'] == 'SINGLE_MATCH').astype(int)
    return features_df

def build_category_features(df):
    features_df = pd.DataFrame()
    features_df['category_HIGHEST_PRIZE'] = (df['category'] == 'HIGHEST_PRIZE').astype(int)
    features_df['category_CHEAPEST'] = (df['category'] == 'CHEAPEST').astype(int)
    return features_df

def build_max_user_lineups_features(df):
    features_df = pd.DataFrame()
    features_df['max_user_lineups_1'] = (df['max_user_lineups'] == 1).astype(int)
    features_df['max_user_lineups_200'] = (df['max_user_lineups'] == 200).astype(int)
    return features_df

def build_championship_group_features(df):
    features_df = pd.DataFrame()
    features_df['championship_group_BR - Serie A'] = (df['championship_group'] == 'BR - Série A').astype(int)
    return features_df

def build_buy_in_features(df):
    features_df = df[['buyin']].copy()
    return features_df

# # type_UNLIMITED
# # room_seasonality_DAILY
# # buyIn
# # max_user_lineups_200
# # category_HIGHEST_PRIZE
# # max_user_lineups_1
# # room_seasonality_SINGLE_MATCH
# # type_LIMITED
# # room_capacity_100
# # category_CHEAPEST
# # championship_group_BR - Série A
def build_features(df: pd.DataFrame, min_max_params_df=None, has_user_id=True):
    feature_functions = [
        build_type_features,
        build_capacity_features,
        build_seasonality_features,
        build_category_features,
        build_max_user_lineups_features,
        build_championship_group_features,
        build_buy_in_features
    ]

    features_df = pd.DataFrame()

    for func in feature_functions:
        feature_df = func(df)
        features_df = pd.concat([features_df, feature_df], axis=1)

    # Normalize
    if min_max_params_df is not None:
        # If min_max_params_df is provided, use it for normalization
        features_df, min_max_params_df = normalize_df(features_df, min_max_params_df)
    else:
        # Otherwise, use the default normalization
        features_df, min_max_params_df = normalize_df(features_df)

    if has_user_id:
        features_df['user_id'] = df['user_id']
    else :
        features_df['room_id'] = df['room_id']


    return features_df, min_max_params_df

def normalize_df(df, min_max_params_df=None):

    if min_max_params_df is not None:
        # If min_max_params_df is provided, use the provided min-max values
        for feature in df.columns:
            min_value = min_max_params_df[min_max_params_df['Feature'] == feature]['Min'].values[0]
            max_value = min_max_params_df[min_max_params_df['Feature'] == feature]['Max'].values[0]
            df[feature] = (df[feature] - min_value) / (max_value - min_value)
    else:
        min_max_params_list = []
        # Calculate and store the min-max parameters for each feature
        for feature in df.columns:
            min_value = df[feature].min()
            max_value = df[feature].max()
            min_max_params_list.append({'Feature': feature, 'Min': min_value, 'Max': max_value})
            df[feature] = (df[feature] - min_value) / (max_value - min_value)

        min_max_params_df = pd.DataFrame(min_max_params_list, columns=['Feature', 'Min', 'Max'])

    return df, min_max_params_df

def filter_users(df):
    user_counts = df['user_id'].value_counts()
    filtered_df = df[df['user_id'].isin(user_counts[user_counts >= 3].index)]
    return filtered_df

def build_users_profiles(users_rooms_with_info):
    users_rooms_with_info = filter_users(users_rooms_with_info)
    user_in_room_with_features, normalization_parameters = build_features(users_rooms_with_info)

    # Aggregate features by user
    user_profiles = user_in_room_with_features.groupby('user_id', as_index=False).mean()

    return user_profiles, normalization_parameters

# RECOMMENDATION ALGORITHM USING USER PROFILE

def get_recommended_rooms_for_users(active_room_features, user_profiles):
    # Extract room_ids and user_ids from the DataFrames
    active_room_ids = active_room_features['room_id']
    user_ids = user_profiles['user_id']

    # Drop the 'room_id' and 'user_id' columns for computation
    active_room_features = active_room_features.drop(columns=['room_id'])
    user_profiles = user_profiles.drop(columns=['user_id'])

    # Convert DataFrames to numpy arrays for faster computation
    active_room_features = active_room_features.values
    user_profiles = user_profiles.values
    cosine_distance_matrix = pairwise_distances(user_profiles, active_room_features, metric='cosine')
    jaccard_distance_matrix = pairwise_distances(user_profiles, active_room_features, metric='jaccard')
    pearson_distance_matrix = pairwise_distances(user_profiles, active_room_features, metric='correlation')



    # Create a DataFrame from the distance matrix
    cosine_similarity_df = 1 - pd.DataFrame(cosine_distance_matrix, index=user_ids, columns=active_room_ids)
    jaccard_similarity_df = 1 - pd.DataFrame(jaccard_distance_matrix, index=user_ids, columns=active_room_ids)
    pearson_similarity_df = 1 - pd.DataFrame(pearson_distance_matrix, index=user_ids, columns=active_room_ids)


    # Create a DataFrame for recommended rooms
    recommended_rooms = []

    for user_id in user_ids:
        # Get recommendations based on cosine similarity
        user_cosine_recommendations = cosine_similarity_df.loc[user_id].reset_index()
        user_cosine_recommendations.columns = ['room_id', 'cosine_similarity']
        user_cosine_recommendations['user_id'] = user_id

        # Make user id to be the second column and cosine the third
        user_cosine_recommendations = user_cosine_recommendations[['user_id', 'room_id', 'cosine_similarity']]

        # Get recommendations based on Jaccard similarity
        user_jaccard_recommendations = jaccard_similarity_df.loc[user_id].reset_index()
        user_jaccard_recommendations.columns = ['room_id', 'jaccard_similarity']

        # Get recommendations based on Pearson similarity
        user_pearson_recommendations = pearson_similarity_df.loc[user_id].reset_index()
        user_pearson_recommendations.columns = ['room_id', 'pearson_similarity']

        # Merge the recommendations from all three similarity measures
        user_recommendations = pd.merge(user_cosine_recommendations, user_jaccard_recommendations, on='room_id')
        user_recommendations = pd.merge(user_recommendations, user_pearson_recommendations, on='room_id')
        # sort by cosine_similarity
        user_recommendations = user_recommendations.sort_values('cosine_similarity', ascending=False)
        recommended_rooms.append(user_recommendations)



    recommended_rooms_df = pd.concat(recommended_rooms)


    return recommended_rooms_df

# Get user profiles created, fetch active rooms from BQ and build recommendations for each user based on similarity
def build_room_recommendations(user_profiles_normalization_parameters, user_profiles, active_rooms_with_info):

    active_room_features, _parameters = build_features(active_rooms_with_info, user_profiles_normalization_parameters, False)

    recommended_rooms = get_recommended_rooms_for_users(active_room_features, user_profiles)

    return recommended_rooms





if __name__ == '__main__':

    user_profiles, normalization_parameters = build_users_profiles(users_rooms_with_info)
    user_profiles.to_csv(USER_PROFILES_FILEPATH)
    normalization_parameters.to_csv(NORMALIZATION_PARAMETERS_FILEPATH)

    selected_user_profiles = user_profiles[user_profiles['user_id'].isin(selected_users['user_id'])]
    recommendations = build_room_recommendations(normalization_parameters, selected_user_profiles, target_rooms_with_info)
    # recommendations.to_csv(USER_ROOM_RECOMMENDATIONS_FILEPATH)
    recommendations.to_csv(USER_ROOM_RECOMMENDATIONS_SELECTED_FILEPATH)
