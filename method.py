# import packages
import pandas as pd
from scipy import stats
from dython import nominal
from dython.nominal import associations

# load the datasets
spotify = pd.read_csv('/spotify_final.csv', index_col=0)
users = pd.read_csv('/non_spotify_final.csv', index_col=0)

# create crosstab dataframes
tabs_spotify = pd.crosstab(spotify.playlist_name, spotify.genre)
tabs_users = pd.crosstab(users.playlist_name, users.genre)

def chi_sq_test(cross_tabs):
    """
    Prints the Chi-Squared Statistic, p-value, and degress of freedom from a Chi-Squared test.

    Args:
        cross_tabs: A crosstab dataframe.
    """
    chi2, p, dof, con_table = stats.chi2_contingency(cross_tabs)
    print(f'chi-squared = {chi2}\np value= {p}\ndegrees of freedom = {dof}')

print('Spotify \n', chi_sq_test(tabs_spotify))
print('\n Users \n', chi_sq_test(tabs_users))

# print cramer's v for spotify playlists and genre
print(nominal.cramers_v(spotify.genre, spotify.playlist_name))
# print cramer's v for user playlists and genre
print(nominal.cramers_v(users.playlist_name, users.genre))

# create and save a correlation heatmap of spotify playlists and genre using cramer's v
associations(spotify[['playlist_name', 'genre']], nom_nom_assoc='cramer', filename= 'cramer_spotify.png', figsize=(10,10))
# create and save a correlation heatmap of user playlists and genre using cramer's v
associations(users[['playlist_name', 'genre']], nom_nom_assoc='cramer', filename= 'cramer_users.png', figsize=(10,10))

# print theil's u for spotify playlists and genre
print(nominal.theils_u(spotify.playlist_name, spotify.genre))
# print theil's u for user playlists and genre
print(nominal.theils_u(users.playlist_name, users.genre))

# create and save a correlation heatmap of spotify playlists and genre using theil's u
associations(spotify[['playlist_name', 'genre']], nom_nom_assoc='theil', filename= 'theil_spotify.png', figsize=(10,10))
# create and save a correlation heatmap of user playlists and genre using theil's u
associations(users[['playlist_name', 'genre']], nom_nom_assoc='theil', filename= 'theil_users.png', figsize=(10,10))
