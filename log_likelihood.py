# import packages
import spacy
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
import unicodedata
import nltk
import re
from collections import Counter
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_venn_wordcloud import venn2_wordcloud

nltk.download('wordnet')
nltk.download('stopwords')

# import multilanguage package from spacy
nlp = spacy.load("xx_sent_ud_sm")

# load the datasets
spotify = pd.read_csv('/Users/jankecoetzee/PycharmProjects/DHthesis/spotify_final.csv', index_col=0)
users = pd.read_csv('/Users/jankecoetzee/PycharmProjects/DHthesis/non_spotify_final.csv', index_col=0)

# create new curator column
spotify['curator'] = 'Spotify'
users['curator'] = 'user'

# merge the spotify and users datasets as a new dataframe
df = pd.concat([spotify, users])

# preprocess the playlist names
playlists = df.playlist_name.values
processed_texts = [text for text in tqdm(nlp.pipe(playlists,
                                              n_process=-1,
                                              disable=["ner",
                                                       "parser"]),
                                          total=len(playlists
                                                   ))]

def clean_text(text):
    punctuation = '!@#$%^&*()_-+={}[]:;\|<>,.?/~`¬°√'
    # import stopwords
    stop_words = set(stopwords.words('english'))

    # Remove punctuation
    cleaned_text = ''.join(character.lower() for character in text if character not in punctuation)

    # Remove accented characters
    cleaned_text = unicodedata.normalize('NFKD', cleaned_text).encode('ASCII', 'ignore').decode('utf-8')

    # Remove possessive apostrophe and trailing "s"
    cleaned_text = re.sub(r"(\w+)'s\b", r'\1', cleaned_text)

    # remove stopwords
    text = ' '.join(word for word in cleaned_text.split() if word.lower() not in stop_words)

    return text

cleaned_texts = [clean_text(item.text) for item in processed_texts]

# add a new column consisting of the cleaned playlist names
df['processed_texts'] = cleaned_texts

# create lists of words that do and do not occur in the playlist names of spotify- and user-curated playlists individually
flatten = lambda t: [item for sublist in t for item in sublist]

subset_spotify = df[(df.curator == 'Spotify')].processed_texts.unique()
subset_spotify = flatten([text.split() for text in subset_spotify])

subset_not_spotify = df[(df.curator != 'Spotify')].processed_texts.unique()
subset_not_spotify = flatten([text.split() for text in subset_not_spotify])

subset_users = df[(df.curator == 'user')].processed_texts.unique()
subset_users = flatten([text.split() for text in subset_users])

subset_not_users = df[(df.curator != 'user')].processed_texts.unique()
subset_not_users = flatten([text.split() for text in subset_not_users])

# function to obtain the distinctive words per playlist
def distinctive_words(target_corpus, reference_corpus):

    stopwords = ['los', 'fra', 'de', 'con', 'g', 'en', 'l']

    counts_c1 = Counter(target_corpus) # count how often each word occurs
    counts_c1 = dict([(key, val) for key, val in counts_c1.items() if key not in stopwords]) # remove useless words
    counts_c2 = Counter(reference_corpus)
    counts_c2 = dict([(key, val) for key, val in counts_c2.items() if key not in stopwords])
    vocabulary = set(list(counts_c1.keys()) + list(counts_c2.keys()))
    freq_c1_total = sum(counts_c1.values())
    freq_c2_total = sum(counts_c2.values())
    results = []
    for word in vocabulary:
        freq_c1 = counts_c1[word]
        freq_c2 = counts_c2[word]
        freq_c1_other = freq_c1_total - freq_c1
        freq_c2_other = freq_c2_total - freq_c2
        llr, p_value,_,_ = chi2_contingency([[freq_c1, freq_c2],
                      [freq_c1_other, freq_c2_other]],
                      lambda_='log-likelihood')
        if freq_c2 / freq_c2_other > freq_c1 / freq_c1_other:
            llr = -llr
        result = {'word':word,
                    'llr':llr,
                    'p_value': p_value}
        results.append(result)
    results_df = pd.DataFrame(results)
    return results_df
  
  # create new dataframes for distinctive words in spotify- and user-curated playlists
results_df_spotify = distinctive_words(subset_spotify, subset_not_spotify)
results_df_spotify = results_df_spotify.sort_values('llr', ascending=False)

results_df_users = distinctive_words(subset_users, subset_not_users)
results_df_users = results_df_users.sort_values('llr', ascending=False)

results_df_spotify['curator'] = 'Spotify'
results_df_users['curator'] = 'users'

results_all = pd.concat([results_df_spotify, results_df_users])
results_all = results_all.sort_values('llr', ascending=False).drop_duplicates(['word'])

top_30_spotify = list(results_all[results_all.curator == 'Spotify'].word.values[:30])
top_30_users = list(results_all[results_all.curator == 'users'].word.values[:30])

word2freq = dict(zip(results_all.word, results_all.llr))

mpl.rcParams['font.family'] = 'DIN Alternate'
%matplotlib inline
fig, ax = plt.subplots(figsize=(10,10), dpi=300)

v = venn2_wordcloud([set(top_30_spotify), set(top_30_users)],
                    ax=ax,
                    set_labels=['Spotify', 'Users'],
                   word_to_frequency=word2freq,
                    wordcloud_kwargs={'font_path': '/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf',
                                     'color_func':lambda *args, **kwargs: (0,0,0)})

v.get_patch_by_id('10').set_color('#27ae60')
v.get_patch_by_id('01').set_color('#3498db')

# Save the figure as an image
plt.savefig('venn_diagram.png')
