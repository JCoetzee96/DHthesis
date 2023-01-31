# import packages
import pandas as pd
import matplotlib.pyplot as plt

# load the datasets
df_500_OG = pd.read_csv('/data/spotify-500.csv')
df_1000_OG = pd.read_csv('/data/non-spotify-1000.csv')

# inspect the datasets
print("Number of unique playlists, Spotify-curated:", df_500_OG.playlist_id.nunique())
print(df_500_OG.info())
print("---------------------")
print("Number of unique playlists, user-curated:", df_1000_OG.playlist_id.nunique())
print(df_1000_OG.info())

# inspect what the dataframe looks like
df_500_OG.head()

# function to create and save a boxplot of the distribution of the total number of songs in a dataframe
def boxplot_total(df, filename):
    plt.figure(figsize=(12,2))
    df.groupby('playlist_id')['track_name'].count().sort_values(ascending=False).plot.box(vert=0);
    plt.yticks([])
    plt.xlabel('Number of songs in a playlist');
    plt.xticks();
    plt.savefig(f'{filename}.png', bbox_inches='tight')
    plt.show()

# function to create and save a boxplot of the distribution of the number of unique songs in a dataframe
def boxplot_unique(df, filename):
    plt.figure(figsize=(12,2))
    df.groupby('playlist_id')['track_name'].nunique().sort_values(ascending=False).plot.box(vert=0);
    plt.yticks([])
    plt.xlabel('Number of unique songs in a playlist');
    plt.xticks();
    plt.savefig(f'{filename}.png', bbox_inches='tight')
    plt.show()

boxplot_total(df_500_OG, 'box_n_songs_500')
boxplot_unique(df_500_OG, 'box_n_unique_songs_500')

print('Spotify-curated playlist \n')
print('Average number of songs per playlist:',round(df_500_OG.groupby(['playlist_id', 'playlist_name'])['track_name'].count().mean()))
print('Average number of unique songs per playlist:',round(df_500_OG.groupby(['playlist_id', 'playlist_name'])['track_name'].nunique().mean()))
df_500_OG.groupby(['playlist_id', 'playlist_name'])['track_name'].nunique().sort_values(ascending=False).reset_index(name='count')

boxplot_total(df_1000_OG, 'box_n_songs_1000')
boxplot_unique(df_1000_OG, 'box_n_unique_songs_1000')

print('User-curated playlist \n')
print('Average number of songs per playlist:',round(df_1000_OG.groupby(['playlist_id', 'playlist_name'])['track_name'].count().mean()))
print('Average number of unique songs per playlist:',round(df_1000_OG.groupby(['playlist_id', 'playlist_name'])['track_name'].nunique().mean()))
df_1000_OG.groupby(['playlist_id', 'playlist_name'])['track_name'].nunique().sort_values(ascending=False).reset_index(name='count')

# create new dataframes with only required features
df_500 = df_500_OG[['playlist_id', 'playlist_name', 'user_id', 'playlist_followers', 'primary_artist_genres']]
df_1000 = df_1000_OG[['playlist_id', 'playlist_name', 'user_id', 'playlist_followers', 'primary_artist_genres']]

df_500.head()

df_1000.head()

# function to create a new DataFrame with the required features, grouped per playlist ID
def playlist_genres(df):
    
    result = {}
    user_ids = []
    followers = []
    
    for idx, row in df.iterrows():
        
        if isinstance(row['primary_artist_genres'], str) and row['primary_artist_genres'].startswith('{'): # only include row if genre column exists and includes a string that starts with '{'
            genres = eval(row['primary_artist_genres'])
            playlist_id = row['playlist_id']
            playlist_name = row['playlist_name']
            user_id = row['user_id']
            n_followers = row['playlist_followers']
            
            if playlist_id not in result:
                result[playlist_id] = {}
                user_ids.append(user_id)
                followers.append(n_followers)
                result[playlist_id][playlist_name] = {}
                result[playlist_id][playlist_name]['genres'] = []
                
            for genre in genres:
                if genre not in result[playlist_id][playlist_name]['genres'] and str(genre) != 'nan':
                    result[playlist_id][playlist_name]['genres'].append(genre)

    playlist_ids = []
    names = []
    g = []
    
    for key, value in result.items(): # key is playlist_id
        playlist_ids.append(key)
        
        for k, v in value.items(): # k = playlist_name, v = genres
            names.append(k)
            for i, j in v.items(): # i = genre, j = list of genres
                g.append(j)
                
    playlist_genre = pd.DataFrame()
    playlist_genre['playlist_id'] = playlist_ids
    playlist_genre['playlist_name'] = names
    playlist_genre['genres'] = g
    playlist_genre['user_id'] = user_ids
    playlist_genre['n_followers'] = followers

    return playlist_genre

df_500_new = playlist_genres(df_500)
df_500_new.head()

df_1000_new = playlist_genres(df_1000)
df_1000_new.head()

print("Number of unique playlists, Spotify-curated:", df_500_new.playlist_id.nunique())
print(df_500_new.info())
print("---------------------")
print("Number of unique playlists, user-curated:", df_1000_new.playlist_id.nunique())
print(df_1000_new.info())

print(df_1000_new.isna().sum() * 100 / len(df_1000_new))
df_1000_new = df_1000_new.dropna()
print(df_1000_new.isna().sum() * 100 / len(df_1000_new)) # remove missing values

# function to create a dictionary, in which the key is the genre and the value the frequency
def counter(df, column):
    count = {}
    for idx, row in df.iterrows():
        values = row[str(column)]
        for value in values:
            if value not in count:
                count[value] = 0
            count[value] += 1
    return count

genre_count_500 = counter(df_500_new, 'genres')
genre_count_1000 = counter(df_1000_new, 'genres')

print('Number of unique genres in Spotify-curated playlists dataset:',len(genre_count_500))
print('Number of unique genres in user-curated playlists dataset:',len(genre_count_1000))
print('\n',genre_count_500)

# function to create and save barplots of a dictionary
def plot_distribution(dict, N, filename):
    myDict = {key:val for key, val in dict.items() if val > N}
    lists = sorted(myDict.items(), key=lambda kv: kv[1], reverse=False)
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.barh(x, y)
    if N == 100:
        plt.xlim(99, 230)
    elif N == 500:
        plt.xlim(475, 850)
    plt.yticks(fontsize = 'xx-small')
    plt.xticks(fontsize = 'x-small')
    plt.xlabel('Frequency', fontsize=7);
    plt.savefig(f'{filename}.png', bbox_inches='tight')
    plt.show()

plot_distribution(genre_count_500, 100, 'genres_500')
plot_distribution(genre_count_1000, 500, 'genres_1000')

# function to obtain the percentages of the key, value in a dictionary within the whole dictionary
def get_percentages(dict):
    perc_dict = {}
    total = sum(dict.values())
    for key,val in dict.items():
        perc_dict[key] = round(val/total * 100, 2)
    return perc_dict

dict_genres_500 = get_percentages(genre_count_500)
percentages_genres_500 = pd.DataFrame(dict_genres_500.items(), columns=["Genre", "Percentage"]).sort_values(by='Percentage', ascending=False)
percentages_genres_500.head(32)

dict_genres_1000 = get_percentages(genre_count_1000)
percentages_genres_1000 = pd.DataFrame(dict_genres_1000.items(), columns=["Genre", "Percentage"]).sort_values(by='Percentage', ascending=False)
percentages_genres_1000.head(32)

# function to count the number of genres within a single value 
def count_columns(df):
    new_df = df.copy()
    count_genres = []
    for idx, row in new_df.iterrows():
        genres = row['genres']
        count_genres.append(len(genres))
    new_df['n_genres'] = count_genres
    new_df = new_df.sort_values(['n_genres'], ascending=False)
    return new_df
    
df_500_counts = count_columns(df_500_new)
df_500_counts.head()

df_1000_counts = count_columns(df_1000_new)
df_1000_counts.head()

print('Spotify-curated playlists:')
print('The average number of genres per playlist:', round(df_500_counts.n_genres.mean()))
print('The average number of followers per playlist:', round(df_500_counts.n_followers.mean()))

print('\nuser-curated playlists:')
print('The average number of genres per playlist:', round(df_1000_counts.n_genres.mean()))
print('The average number of followers per playlist:', round(df_1000_counts.n_followers.mean()))

# create random sample of the user-curated playlists
df_1000_final = df_1000_new.sample(n=486, random_state=52)
df_1000_final.info()

df_1000_final.head()

# function to split the genres over rows according to the corresponding playlist
def split_sets(df):
    gen = df['genres'].apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 'value']].set_index('index')
    new_df = gen.merge(df_500_new[['playlist_id', 'playlist_name']], left_index=True, right_index=True, how='right').rename(columns={'value':'genre'})
    return new_df

df_500_finalfinal = split_sets(df_500_new)
df_500_finalfinal.head()

df_1000_finalfinal = split_sets(df_1000_final)
df_1000_finalfinal.head()

# save the newly created datasets
df_500_finalfinal.to_csv('spotify_final.csv')
df_1000_finalfinal.to_csv('non_spotify_final.csv')
