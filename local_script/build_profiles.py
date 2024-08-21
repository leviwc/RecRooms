import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt



USER_IN_ROOMS = 'user_in_room_last_30_days_with_info.csv'
USER_PROFILES_FILEPATH = 'user_profiles_based_on_last_30_days.csv'
NORMALIZATION_PARAMETERS_FILEPATH = 'user_profiles_normalization_parameters.csv'
ROOMS_WITH_INFO_FILEPATH = 'rooms_in_last_30_days_with_info.csv'
SELECTED_ROOMS_WITH_INFO_FILEPATH = 'selected/selected_rooms_with_info.csv'
USER_ROOM_RECOMMENDATIONS_FILEPATH = 'user_room_recommendations.csv'
USER_ROOM_RECOMMENDATIONS_SELECTED_FILEPATH = 'selected/user_room_recommendations_selected.csv'
SELECTED_USERS_WITH_ROOMS = 'selected/selected_users_with_rooms.csv'
SELECTED_USERS = 'selected/selected_users.csv'
MRR_FILEPATH = 'selected/mrr.csv'

users_rooms_with_info = pd.read_csv(USER_IN_ROOMS)
user_id_with_room_ids = users_rooms_with_info[['user_id', 'room_id']]

selected_target_rooms_with_info = pd.read_csv(SELECTED_ROOMS_WITH_INFO_FILEPATH)
selected_users = pd.read_csv(SELECTED_USERS)

relevancy_grade_df = pd.read_csv(SELECTED_USERS_WITH_ROOMS)
relevancy_grade_df['relevancy_grade'] = 1

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
    features_df.fillna(0, inplace=True)

    return features_df, min_max_params_df

def normalize_df(df, min_max_params_df=None):

    if min_max_params_df is not None:
        # If min_max_params_df is provided, use the provided min-max values
        for feature in df.columns:
            min_value = min_max_params_df[min_max_params_df['Feature'] == feature]['Min'].values[0]
            max_value = min_max_params_df[min_max_params_df['Feature'] == feature]['Max'].values[0]
            if max_value == min_value:
                df[feature] = 0
            else:
                df[feature] = (df[feature] - min_value) / (max_value - min_value)
    else:
        min_max_params_list = []
        # Calculate and store the min-max parameters for each feature
        for feature in df.columns:
            min_value = df[feature].min()
            max_value = df[feature].max()
            min_max_params_list.append({'Feature': feature, 'Min': min_value, 'Max': max_value})
            if max_value == min_value:
                df[feature] = 0
            else:
                df[feature] = (df[feature] - min_value) / (max_value - min_value)

        min_max_params_df = pd.DataFrame(min_max_params_list, columns=['Feature', 'Min', 'Max'])

    return df, min_max_params_df

def filter_users(df, amount_of_entries):
    # assure that the entries will be in different rooms:
    user_counts = df.drop_duplicates(subset=['user_id', 'room_id']).groupby('user_id').size()

    filtered_df = df[df['user_id'].isin(user_counts[user_counts >= amount_of_entries].index)]
    return filtered_df

def build_users_profiles(users_rooms_with_info):
    users_rooms_with_info = filter_users(users_rooms_with_info, 3)
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
    cosine_distance_matrix = cdist(user_profiles, active_room_features, metric='cosine')
    jaccard_distance_matrix = cdist(user_profiles, active_room_features, metric='jaccard')
    pearson_distance_matrix = cdist(user_profiles, active_room_features, metric='correlation')



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
        user_recommendations = pd.merge(user_cosine_recommendations, user_jaccard_recommendations, on='room_id', how='outer')
        user_recommendations = pd.merge(user_recommendations, user_pearson_recommendations, on='room_id', how='outer')

        # Calculate a rank for each similarity score for the user, and add it in a new column for each of them
        user_recommendations['cosine_similarity_rank'] = user_recommendations['cosine_similarity'].rank(ascending=False)
        user_recommendations['jaccard_similarity_rank'] = user_recommendations['jaccard_similarity'].rank(ascending=False)
        user_recommendations['pearson_similarity_rank'] = user_recommendations['pearson_similarity'].rank(ascending=False)
        recommended_rooms.append(user_recommendations)



    recommended_rooms_df = pd.concat(recommended_rooms, ignore_index=True)


    return recommended_rooms_df

# Get user profiles created, fetch active rooms from BQ and build recommendations for each user based on similarity
def build_room_recommendations(user_profiles_normalization_parameters, user_profiles, active_rooms_with_info):

    active_room_features, _parameters = build_features(active_rooms_with_info, user_profiles_normalization_parameters, False)

    recommended_rooms = get_recommended_rooms_for_users(active_room_features, user_profiles)

    return recommended_rooms


def calculate_mrr_with_relevancy_grade(recommendations, relevancy_grade_df):
    # Merge the recommendations with the relevancy grade
    recommendations_with_relevancy_grade = pd.merge(recommendations, relevancy_grade_df, on=['user_id', 'room_id'], how='left').fillna(0)

    relevances_rank = recommendations_with_relevancy_grade.groupby(['user_id', 'relevancy_grade'])['cosine_similarity_rank'].min()
    ranks = relevances_rank.loc[:, 1]
    reciprocal_ranks = 1 / (ranks)

    mrr_cosine_similarity = reciprocal_ranks.mean()

    relevances_rank = recommendations_with_relevancy_grade.groupby(['user_id', 'relevancy_grade'])['jaccard_similarity_rank'].min()
    ranks = relevances_rank.loc[:, 1]
    reciprocal_ranks = 1 / (ranks)
    mrr_jaccard_similarity = reciprocal_ranks.mean()

    relevances_rank = recommendations_with_relevancy_grade.groupby(['user_id', 'relevancy_grade'])['pearson_similarity_rank'].min()
    ranks = relevances_rank.loc[:, 1]
    reciprocal_ranks = 1 / (ranks)
    mrr_pearson_similarity = reciprocal_ranks.mean()

    # make a df with the results
    mrr_df = pd.DataFrame({'cosine_similarity': [mrr_cosine_similarity], 'jaccard_similarity': [mrr_jaccard_similarity], 'pearson_similarity': [mrr_pearson_similarity]})

    return mrr_df


def calculate_ap_for_given_distance(recommendation_df, relevant_rooms, chosen_distance, rank):
    relevant_rooms['relevancy'] = 1
    recommendation_df = pd.merge(recommendation_df, relevant_rooms, on='room_id', how='left').fillna(0)
    # new df with only relevancy and {chosen_distance}_similarity_rank
    recommendation_df = recommendation_df[['relevancy', f'{chosen_distance}_similarity_rank']]
    recommendation_df = recommendation_df.sort_values(by=f'{chosen_distance}_similarity_rank')
    cumsum = recommendation_df['relevancy'].cumsum()
    for i in range(1, len(cumsum) + 1):
        # check if index is relevant
        if recommendation_df.iloc[i - 1]['relevancy'] == 0 or i > rank:
            cumsum.iloc[i - 1] = 0
        else:
            cumsum.iloc[i - 1] = cumsum.iloc[i - 1] / i
    average_precision = cumsum.sum() / len(relevant_rooms)
    df = pd.DataFrame({'method': [chosen_distance], 'average_precision': [average_precision], 'rank': [rank]})
    return df

def calculate_ap_for_recommendation(recommendation, relevant_rooms, rank):
    # Cosine
    cosine_df = calculate_ap_for_given_distance(recommendation, relevant_rooms, 'cosine', rank)
    # Jaccard
    jaccard_df = calculate_ap_for_given_distance(recommendation, relevant_rooms, 'jaccard', rank)
    # Pearson
    pearson_df = calculate_ap_for_given_distance(recommendation, relevant_rooms, 'pearson', rank)
    ans_df = pd.concat([cosine_df, jaccard_df, pearson_df])
    return ans_df

def automatic_avaliation(user_rooms_with_info, relevant_sample_size, random_irrelevant_sample_size, minimum_entried_rooms_of_user, ranks):
    user_rooms_with_info_filtered = filter_users(user_rooms_with_info, minimum_entried_rooms_of_user)
    ranks = [3, 5, 10]



    # group by room id and remove user ids
    all_rooms_with_info = user_rooms_with_info_filtered
    all_rooms_with_info = all_rooms_with_info.drop(columns=['user_id']).drop_duplicates()
    grouped_user_df = user_rooms_with_info_filtered.groupby('user_id')

    ans_list = []
    for user_id, user_df in grouped_user_df:
        #get up to 5 random rooms from user df with different room ids, even tho user_df has duplicate room ids
        separate_rooms_df = user_df.drop_duplicates(subset='room_id').sample(n=relevant_sample_size, random_state=42)

        #print number of rows if number of rows is not 35
        if len(separate_rooms_df) != relevant_sample_size:
            separate_rooms_df.to_csv('this_is_weird_separated_rooms.csv')
        # drop the rows with room id in separate_rooms_df
        remainings_df = user_df.drop(user_df[user_df['room_id'].isin(separate_rooms_df['room_id'])].index)
        # remainings_df = remainings_df.drop(columns=['room_id'])
        user_profile, parameters = build_users_profiles(remainings_df)
        random_rooms = all_rooms_with_info[~all_rooms_with_info['room_id'].isin(user_df['room_id'])].sample(n=random_irrelevant_sample_size, random_state=42)
        if len(random_rooms) != random_irrelevant_sample_size:
            random_rooms.to_csv('this_is_weird_random_rooms.csv')
        random_rooms = pd.merge(random_rooms, separate_rooms_df, how='outer')
        if len(random_rooms) != random_irrelevant_sample_size + relevant_sample_size:
            random_rooms.to_csv('this_is_weird_merged_random_rooms.csv')
        random_rooms_with_features, _parameters = build_features(random_rooms, parameters, False)
        if len(random_rooms_with_features) != random_irrelevant_sample_size + relevant_sample_size:
            random_rooms_with_features.to_csv('weird_weird_random_rooms_with_features.csv')
        recommendations_df = get_recommended_rooms_for_users(random_rooms_with_features, user_profile)
        if len(recommendations_df) != random_irrelevant_sample_size + relevant_sample_size:
            recommendations_df.to_csv('weird_weird_recommendations_df.csv')
        # calculate ap for each rank
        for rank in ranks:
            ap_calculation = calculate_ap_for_recommendation(recommendations_df, separate_rooms_df, rank)
            ans_list.append(ap_calculation)
    ans_df = pd.concat(ans_list).groupby(['method', 'rank']).mean()
    return ans_df



if __name__ == '__main__':

    # user_profiles, normalization_parameters = build_users_profiles(users_rooms_with_info)
    # user_profiles.to_csv(USER_PROFILES_FILEPATH)
    # normalization_parameters.to_csv(NORMALIZATION_PARAMETERS_FILEPATH)

    # selected_user_profiles = user_profiles[user_profiles['user_id'].isin(selected_users['user_id'])]
    # recommendations = build_room_recommendations(normalization_parameters, selected_user_profiles, target_rooms_with_info)
    # # recommendations.to_csv(USER_ROOM_RECOMMENDATIONS_FILEPATH)
    # recommendations.to_csv(USER_ROOM_RECOMMENDATIONS_SELECTED_FILEPATH)
    # calculate_mrr_with_relevancy_grade(recommendations, relevancy_grade_df)
    # mrr_df = calculate_mrr_with_relevancy_grade(recommendations, relevancy_grade_df)
    # mrr_df.to_csv(MRR_FILEPATH)

    # relevant_sample_size = 1
    # random_irrelevant_sample_size = 24
    # minimum_entried_rooms_of_user = 10
    # ranks = [3, 5, 10]
    # ans_df = automatic_avaliation(users_rooms_with_info, relevant_sample_size, random_irrelevant_sample_size, minimum_entried_rooms_of_user, ranks)
    # print(ans_df)
    # ans_df.to_csv('ans_1_relevant_24_irrelevant_2.csv')
    ans_df = pd.read_csv('ans_1_relevant_24_irrelevant_2.csv')

    # make a bars graph with y axis as the average precision and x axis as the rank, and colored by the method and convert average precision to percentage, and show % in the graph y axis
    ans_df['average_precision'] = ans_df['average_precision'] * 100
    ans_df.pivot(index='rank', columns='method', values='average_precision').plot.bar()
    # make the graph but prettier
    plt.xlabel('Rank')
    plt.ylabel('Precision')
    plt.title('Mean Reciprocal Precision Rank (MRR)')
    # Fix the numbers rotation on the x axis
    plt.xticks(rotation=0)
    # Add the % in the y axis numbers
    plt.gca().yaxis.set_major_formatter('{:.0f}%'.format)
    #write the graph to  a file
    plt.savefig('mean_recriprocal_rank_for_each_rank_2.png')
    #render
    plt.show()



