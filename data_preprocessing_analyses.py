# import packages
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice
from more_itertools import unique_everseen
import string

# load the datasets
df_500_OG = pd.read_csv('//data/spotify-500.csv')
df_1000_OG = pd.read_csv('/data/non-spotify-1000.csv')

# inspect the datasets
print("Number of unique playlists, Spotify-curated:", df_500_OG.playlist_id.nunique())
print(df_500_OG.info())
print("---------------------")
print("Number of unique playlists, user-curated:", df_1000_OG.playlist_id.nunique())
print(df_1000_OG.info())

# inspect what the dataframe looks like
df_500_OG.head()

# create a new dataframe consisting of 500 random unique playlists
unique_playlists = list(islice(unique_everseen(df_1000_OG.playlist_id), 500))
df_1000_sample = df_1000_OG.query('playlist_id in @unique_playlists')

print(df_1000_sample.playlist_id.nunique()) # check that there are 500 playlists in the newly created DataFrame
df_1000_sample.info() # inspect the data

# create a boxplot of the distribution of the number of songs per playlist
def boxplot_total(df, filename, format):
    plt.figure(figsize=(12,2))
    if format == 'original':
        df.groupby('playlist_id')['track_name'].count().sort_values(ascending=False).plot.box(vert=0);
    elif format == 'unique':
        df.groupby('playlist_id')['track_name'].nunique().sort_values(ascending=False).plot.box(vert=0);
    plt.yticks([])
    plt.xlabel('Number of songs in a playlist');
    plt.xticks();
    plt.savefig(f'{filename}.png', bbox_inches='tight')
    plt.show()

boxplot_total(df_500_OG, 'box_n_songs_500', 'original')
boxplot_total(df_500_OG, 'box_n_unique_songs_500', 'unique')

print('Spotify-curated playlist \n')
print('Average number of songs per playlist:',round(df_500_OG.groupby(['playlist_id', 'playlist_name'])['track_name'].count().mean()))
print('Average number of unique songs per playlist:',round(df_500_OG.groupby(['playlist_id', 'playlist_name'])['track_name'].nunique().mean()))
# check the playlists with the most number of songs
df_500_OG.groupby(['playlist_id', 'playlist_name'])['track_name'].nunique().sort_values(ascending=False).reset_index(name='count')

boxplot_total(df_1000_sample, 'box_n_songs_1000', 'original')
boxplot_total(df_1000_sample, 'box_n_unique_songs_1000', 'unique')

print('User-curated playlist \n')
print('Average number of songs per playlist:',round(df_1000_sample.groupby(['playlist_id', 'playlist_name'])['track_name'].count().mean()))
print('Average number of unique songs per playlist:',round(df_1000_sample.groupby(['playlist_id', 'playlist_name'])['track_name'].nunique().mean()))
df_1000_sample.groupby(['playlist_id', 'playlist_name'])['track_name'].nunique().sort_values(ascending=False).reset_index(name='count')

# create new dataframes with only required features
df_500 = df_500_OG[['playlist_id', 'playlist_name', 'user_id', 'playlist_followers', 'primary_artist_genres']]
df_1000 = df_1000_sample[['playlist_id', 'playlist_name', 'user_id', 'playlist_followers', 'primary_artist_genres']]

df_500.head()

df_1000.head()

# create a new dataframe in which the genre column values are transformed to lists
def playlist_genres(df):
    result = {}
    user_ids = []
    followers = []
    for idx, row in df.iterrows():
        if isinstance(row['primary_artist_genres'], str) and row['primary_artist_genres'].startswith('{'): # only include row information if the genre column value is not empty
            genres = eval(row['primary_artist_genres'])
            playlist_id = row['playlist_id']
            playlist_name = row['playlist_name']
            user_id = row['user_id']
            n_followers = row['playlist_followers']
            if playlist_id not in result: # prevent duplicates of the playlist_id
                result[playlist_id] = {}
                user_ids.append(user_id)
                followers.append(n_followers)
                result[playlist_id][playlist_name] = {}
                result[playlist_id][playlist_name]['genres'] = []
            for genre in genres: # prevent duplicates per playlist
                if genre not in result[playlist_id][playlist_name]['genres'] and str(genre) != 'nan':
                    result[playlist_id][playlist_name]['genres'].append(genre)

    playlist_ids = []
    names = []
    g = []
    for key, value in result.items():
        playlist_ids.append(key)
        for k, v in value.items(): # k = playlist_name, v = genres & countries
            names.append(k)
            for i, j in v.items(): # i = genre, j = list of genres
                g.append(j)
#
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

# check whether there are any missing values in the newly created dataframes
print(df_500_new.isna().sum() * 100 / len(df_1000_new))
print(df_1000_new.isna().sum() * 100 / len(df_1000_new))

# create random sample of the user-curated playlists
df_1000_adjusted = df_1000_new.sample(n=486, random_state=52)
df_1000_adjusted.info()

# create dictionary in which the genre is the key and the value is the frequency
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
genre_count_1000 = counter(df_1000_adjusted, 'genres')

print('Number of unique genres in Spotify-curated playlists dataset:',len(genre_count_500))
print('Number of unique genres in user-curated playlists dataset:',len(genre_count_1000))
print('\n',genre_count_500)

# plot the distribution of genres
def plot_distribution(dict, N, filename):
    myDict = {key:val for key, val in dict.items() if val > N}
    lists = sorted(myDict.items(), key=lambda kv: kv[1], reverse=False)
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.barh(x, y)
    plt.xlim(99, 430)
    plt.yticks(fontsize = 'xx-small')
    plt.xticks(fontsize = 'x-small')
    plt.xlabel('Frequency', fontsize=7);
    plt.savefig(f'{filename}.png', bbox_inches='tight')
    plt.show()

plot_distribution(genre_count_500, 100, 'genres_500')
plot_distribution(genre_count_1000, 245, 'genres_1000')

# get the percentage of how often the genre occurs
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

# function to lowercase and remove punctuations
def clean_text(text):
    text = ' '.join(character.lower() for character in text)
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def clean_data(df):
    new_df = df.copy()
    new_df['playlist_name'] = new_df['playlist_name'].str.split(' ')
    new_df['playlist_name'] = new_df['playlist_name'].apply(clean_text)
    return new_df

def genre_playlist(df):
    count = 0
    for idx, row in df.iterrows():
        genres = row['genres']
        playlist_name = row['playlist_name']
        for genre in genres:
            if genre in playlist_name:
                count += 1
    return count

clean_500 = clean_data(df_500_new)
clean_1000 = clean_data(df_1000_adjusted)

print('Number of rows in the cleaned Spotify dataset:',clean_500.shape[0])
print('Number of times the a genre of the playlist is mentioned in the playlist name:',
      genre_playlist(clean_500))
print(round(genre_playlist(clean_500) / clean_500.shape[0], 2))

print('\nNumber of rows in the cleaned user-curated playlists dataset:',clean_1000.shape[0])
print('Number of times the a genre of the playlist is mentioned in the playlist name:',
      genre_playlist(clean_1000))
print(round(genre_playlist(clean_1000) / clean_1000.shape[0], 2))

# count how many genres there are per playlist
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

df_1000_counts = count_columns(df_1000_adjusted)
df_1000_counts.head()

print('Spotify-curated playlists:')
print('The average number of genres per playlist:', round(df_500_counts.n_genres.mean()))
print('The average number of followers per playlist:', round(df_500_counts.n_followers.mean()))

print('\nuser-curated playlists:')
print('The average number of genres per playlist:', round(df_1000_counts.n_genres.mean()))
print('The average number of followers per playlist:', round(df_1000_counts.n_followers.mean()))

# create new dataframe in which each row represents an individual (unique) genre per playlist
def split_sets(df):
    gen = df['genres'].apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 'value']].set_index('index')
    new_df = gen.merge(df_500_new[['playlist_id', 'playlist_name']], left_index=True, right_index=True, how='right').rename(columns={'value':'genre'})
    return new_df

df_500_final = split_sets(df_500_new)
df_500_final.head()

df_1000_final = split_sets(df_1000_new)
df_1000_final.head()

# save the results as new csv files
df_500_final.to_csv('spotify_final.csv')
df_1000_final.to_csv('non_spotify_final.csv')
