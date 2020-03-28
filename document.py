from os import path
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from helpers import create_graph_of_words


class Document:
    """
    This is class is for representing a document from the CS276 dataset
    """

    def __init__(self, id_doc: int, url: str, folder: int):
        self.__id = id_doc
        self.__folder = folder
        self.__url = url
        self.tokens = []
        self.__key_words = []
        self.__length = 0

    @property
    def length(self):
        return self.__length

    @property
    def id(self):
        return self.__id

    @property
    def folder(self):
        return self.__folder

    @property
    def url(self):
        return self.__url

    @property
    def key_words(self):
        return self.__key_words

    def load_data(self, path_to_documents):
        path_to_file = path.join(
            path_to_documents, "{}/{}".format(self.__folder, self.__url)
        )
        with open(path_to_file, "r") as file:
            for line in file.readlines():
                self.tokens.extend(line.rstrip("\n").split(" "))
        self.__remove_not_alpha()

    def __remove_not_alpha(self):
        filtered_tokens = []
        for token in self.tokens:
            if token.isalpha():
                filtered_tokens.append(token)
        self.tokens = filtered_tokens
        self.__length = len(self.tokens)

    def __store_key_words(self):
        counter = Counter(self.tokens)
        self.__key_words = [x[0] for x in counter.most_common(5)]

    def __remove_stopwords(self, stopwords_list):
        self.tokens = [token for token in self.tokens if token not in stopwords_list]
        self.__length = len(self.tokens)

    def __lemmatize(self, lemmatizer):
        self.tokens = [lemmatizer.lemmatize(token) for token in self.tokens]

    def get_term_weights(self):
        graph_of_words = create_graph_of_words(window=4, tokens=self.tokens)
        term_weights = {}
        for term, indegree_edges in graph_of_words.items():
            term_weights[term] = len(indegree_edges)
        return term_weights

    def process_document(self, stopwords_list, lemmatizer):
        self.__remove_stopwords(stopwords_list)
        self.__store_key_words()
        self.__lemmatize(lemmatizer)

    def get_vocabulary(self):
        return list(set(self.tokens))


if __name__ == "__main__":
    document = Document(url="3dradiology.stanford.edu_", folder=0, id_doc=0)
    document.load_data("data/cs276")
    word_net_lemmatizer = WordNetLemmatizer()
    nltk_stopwords = stopwords.words("english")
    document.process_document(nltk_stopwords, word_net_lemmatizer)
