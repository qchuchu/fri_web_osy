import cProfile

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from search_engine import SearchEngine

nltk_stopwords = stopwords.words("english")
word_net_lemmatizer = WordNetLemmatizer()
search_engine = SearchEngine(
    collection_name="cs276",
    stopwords_list=nltk_stopwords,
    lemmatizer=word_net_lemmatizer,
)

cProfile.run("search_engine.search('stanford class')")
