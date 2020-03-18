from os import path, getcwd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer


class Document:
    """
    This class is for representing a document from the CS276 dataset
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

    def load_data(self, path_to_documents):
        """

        :param path_to_documents: path to the collection downloaded from http://web.stanford.edu/class/cs276/pa/pa1-data.zip

        Open and read the document
        Set the private attribute tokens as a list of words presents in the document
        """
        path_to_file = path.join(
            path_to_documents,
            "{}/{}".format(self.__folder, self.__url))
        with open(path_to_file, 'r') as file:
            for line in file.readlines():
                self.tokens.extend(line.rstrip('\n').split(' '))
        self.__remove_not_alpha()
        self.__store_key_words()

    def __remove_not_alpha(self):
        """
            Remove the tokens with non alpha characters of the tokens attribute.
            Updates the length attribute.
        """
        filtered_tokens = []
        for token in self.tokens:
            if token.isalpha():
                filtered_tokens.append(token)
        self.tokens = filtered_tokens
        self.__length = len(self.tokens)

    def __store_key_words(self):
        """
        Stores a list of the 5 most common words of the document
        """
        counter = Counter(self.tokens)
        self.__key_words = [x[0] for x in counter.most_common(5)]

    def __remove_stopwords(self, stopwords_list):
        """
        Remove the useless words that are gathered in the stopwords collection of nltk
        """
        self.tokens = [token for token in self.tokens if token not in stopwords_list]
        self.__length = len(self.tokens)

    def __lemmatize(self, lemmatizer):
        self.tokens = [lemmatizer.lemmatize(token) for token in self.tokens]

    def __create_graph_of_words(self, window):
        """

        :param window : int, size of the sliding window used to create the graph
        :return: an undirected graph of words for the document
        """
        n = len(self.tokens)
        graph_of_words = {}
        for i in range(n):
            current_word = self.tokens[i]
            for j in range(i + 1, min(i + window, n)):
                observed_word = self.tokens[j]
                if observed_word in graph_of_words:
                    graph_of_words[observed_word].update([current_word])
                else:
                    graph_of_words[observed_word] = {current_word}
        return graph_of_words

    def get_term_weights(self):
        graph_of_words = self.__create_graph_of_words(window=4)
        term_weights = {}
        for term, indegree_edges in graph_of_words.items():
            term_weights[term] = len(indegree_edges)
        return term_weights

    def process_document(self, stopwords_list, lemmatizer):
        self.__remove_stopwords(stopwords_list)
        self.__lemmatize(lemmatizer)


if __name__ == '__main__':
    document = Document(url="3dradiology.stanford.edu_", folder=0, id_doc=0)
    document.load_data("data/cs276")
    word_net_lemmatizer = WordNetLemmatizer()
    nltk_stopwords = stopwords.words('english')
    document.process_document(nltk_stopwords, word_net_lemmatizer)
    document.get_term_weights()

