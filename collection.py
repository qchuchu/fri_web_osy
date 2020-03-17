from os import path, listdir, getcwd, walk
import pickle
from document import Document
from math import log
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple


class Collection:
    """
    This class will represent a collection of documents
    """

    def __init__(self, name, stopwords_list, lemmatizer):
        """

        :param name: Chosen name for the collection
        :param stopwords_list: list of words that won't be taken into account.
        :param lemmatizer: Method to lemmatize / stem the document.

        """
        self.__name = name
        self.documents: List[Document] = []
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.stopwords = stopwords_list
        self.lemmatizer = lemmatizer
        self.path_to_data = path.join(getcwd(), "data/{}".format(name))
        self.nb_docs = sum([len(files) for r, d, files in walk(self.path_to_data)])
        self.average_document_length = 0
        print("Loading Documents...")
        self.__load_documents()
        print("All Document Loaded !")
        print("Loading Inverted Index...")
        self.__load_inverted_index()
        print("Inverted Index Loaded")

    @property
    def name(self):
        return self.__name

    def __load_documents(self):
        """Load each document in a vector of processed
        tokens and serialize the vectors in a .p file, if this file doesn't already exist.
        Assign the vectors to self.documents.
        """
        try:
            filename = "{}_preprocessed_documents.p".format(self.name)
            self.documents = pickle.load(open(filename, "rb"))
        except FileNotFoundError as e:
            nb_document_loaded = 0
            for directory_index in range(10):
                path_directory = path.join(self.path_to_data, str(directory_index))
                for filename in listdir(path_directory):
                    document = Document(url=filename, folder=directory_index, id_doc=nb_document_loaded)
                    document.load_data(self.path_to_data)
                    document.process_document(
                        stopwords_list=self.stopwords,
                        lemmatizer=self.lemmatizer
                    )
                    self.documents.append(document)
                    nb_document_loaded += 1
                    print("{}/{} document loaded !".format(nb_document_loaded, self.nb_docs))
            self.__store_processed_documents()
        assert len(self.documents) == self.nb_docs
        for document in self.documents:
            self.average_document_length += document.length
        self.average_document_length /= self.nb_docs

    def __store_processed_documents(self):
        """Serialize the tokenized self.documents using pickle"""
        preprocessed_documents = open("{}_preprocessed_documents.p".format(self.name), "wb")
        pickle.dump(self.documents, preprocessed_documents)

    def __load_inverted_index(self):
        """
        Create an inverted index with term weights per document, if it doesn't exist already
        """
        try:
            filename = "{}_inverted_index.p".format(self.name)
            self.inverted_index = pickle.load(open(filename, "rb"))
        except FileNotFoundError as e:
            for document in self.documents:
                term_weights = document.get_term_weights()
                for term, weight in term_weights.items():
                    if term in self.inverted_index:
                        self.inverted_index[term].append((document.id, weight))
                    else:
                        self.inverted_index[term] = [(document.id, weight)]
            self.__store_inverted_index()

    def __store_inverted_index(self):
        inverted_index_file = open("{}_inverted_index.p".format(self.name), "wb")
        pickle.dump(self.inverted_index, inverted_index_file)

    def get_vocabulary(self):
        """
            Scan the inverted index and return a list of all the terms in the index.
        """
        return list(self.inverted_index.keys())

    def __get_term_weight(self, target_term, target_doc_id):
        """
        :return: Return 0 if the term is not in the inverted index or the document doesn't comprise
        this term, otherwise returns the weight of target_term in target_doc_id.
        """
        try:
            term_weights = self.inverted_index[target_term]
        except KeyError as e:
            return 0
        for doc_id, weight in term_weights:
            if doc_id == target_doc_id:
                return weight
        return 0

    def __get_pivoted_term_weight(self, target_term, target_doc_id, b):
        """
             Calculate the term weigth normalized by the length of the document,
             using the tw-idf method.
        """
        term_weight = self.__get_term_weight(target_term, target_doc_id)
        if term_weight == 0:
            return 0
        pivoted_normalizer = 1 - b + (b * self.documents[target_doc_id].length / self.average_document_length)
        return term_weight / pivoted_normalizer

    def __get_idf(self, target_term):
        try:
            df = len(self.inverted_index[target_term])
        except KeyError as e:
            return 0
        return log((self.nb_docs + 1) / df)

    def get_tw_idf(self, target_term, target_doc_id, b):
        return self.__get_pivoted_term_weight(target_term, target_doc_id, b) * self.__get_idf(target_term)


if __name__ == '__main__':
    word_net_lemmatizer = WordNetLemmatizer()
    nltk_stopwords = stopwords.words('english')
    collection = Collection(name='cs276', stopwords_list=nltk_stopwords, lemmatizer=word_net_lemmatizer)
    print(collection.get_tw_idf(
        target_term='fra',
        target_doc_id=14,
        b=0.003
    ))
