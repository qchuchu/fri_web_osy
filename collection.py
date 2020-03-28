from os import path, listdir, getcwd, walk
from pickle import load, dump
from typing import List, Dict, Tuple
from math import log, sqrt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from document import Document


class Collection:
    """
    This class will represent a collection of documents
    """

    def __init__(self, name: str, stopwords_list: List[str], lemmatizer):
        self.__name = name
        self.documents: List[Document] = []
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.documents_norms: Dict[int, float] = {}
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
        print("Load Documents Norms...")
        self.__load_documents_norms()
        print("Documents Norms Loaded...")

    @property
    def name(self):
        return self.__name

    def __load_documents(self):
        try:
            self.documents = self.__load_pickle_file("preprocessed_documents")
        except FileNotFoundError:
            nb_document_loaded = 0
            for directory_index in range(10):
                print("Processing folder #{}...".format(directory_index))
                path_directory = path.join(self.path_to_data, str(directory_index))
                for filename in tqdm(listdir(path_directory)):
                    document = Document(
                        url=filename, folder=directory_index, id_doc=nb_document_loaded
                    )
                    document.load_data(self.path_to_data)
                    document.process_document(
                        stopwords_list=self.stopwords, lemmatizer=self.lemmatizer
                    )
                    self.documents.append(document)
                    nb_document_loaded += 1
            self.__store_pickle_file("preprocessed_documents", self.documents)
        assert len(self.documents) == self.nb_docs
        for document in self.documents:
            self.average_document_length += document.length
        self.average_document_length /= self.nb_docs

    def __load_inverted_index(self):
        try:
            self.inverted_index = self.__load_pickle_file("inverted_index")
        except FileNotFoundError:
            for document in self.documents:
                term_weights = document.get_term_weights()
                for term, weight in term_weights.items():
                    if term in self.inverted_index:
                        self.inverted_index[term].append((document.id, weight))
                    else:
                        self.inverted_index[term] = [(document.id, weight)]
            self.__store_pickle_file("inverted_index", self.inverted_index)

    def __load_documents_norms(self):
        try:
            self.documents_norms = self.__load_pickle_file("documents_norms")
        except FileNotFoundError:
            nb_norms_calculated = 0
            for document in self.documents:
                doc_vocabulary = document.get_vocabulary()
                norm = 0
                for token in doc_vocabulary:
                    norm += self.get_tw_idf(token, document.id, 0.003) ** 2
                norm = sqrt(norm)
                self.documents_norms[document.id] = norm
                nb_norms_calculated += 1
                print(
                    "{}/{} norms calculated !".format(nb_norms_calculated, self.nb_docs)
                )
            self.__store_pickle_file("documents_norms", self.documents_norms)

    def __load_pickle_file(self, filename):
        pickle_filename = "{}_{}.p".format(self.name, filename)
        return load(open(pickle_filename, "rb"))

    def __store_pickle_file(self, filename, collection_object):
        target_file = open("{}_{}.p".format(self.name, filename), "wb")
        dump(collection_object, target_file)

    def get_vocabulary(self):
        return list(self.inverted_index.keys())

    def __get_term_weight(self, target_term, target_doc_id):
        try:
            term_weights = self.inverted_index[target_term]
        except KeyError:
            return 0
        for doc_id, weight in term_weights:
            if doc_id == target_doc_id:
                return weight
        return 0

    def __get_pivoted_term_weight(self, target_term, target_doc_id, b):
        term_weight = self.__get_term_weight(target_term, target_doc_id)
        if term_weight == 0:
            return 0
        pivoted_normalizer = (
            1
            - b
            + (b * self.documents[target_doc_id].length / self.average_document_length)
        )
        return term_weight / pivoted_normalizer

    def get_idf(self, target_term):
        try:
            df = len(self.inverted_index[target_term])
        except KeyError:
            return 0
        return log((self.nb_docs + 1) / df)

    def get_tw_idf(self, target_term, target_doc_id, b):
        return self.__get_pivoted_term_weight(
            target_term, target_doc_id, b
        ) * self.get_idf(target_term)

    def get_posting_list(self, target_term):
        try:
            doc_list = [doc_id for doc_id, weight in self.inverted_index[target_term]]
        except KeyError:
            return []
        return doc_list


if __name__ == "__main__":
    word_net_lemmatizer = WordNetLemmatizer()
    nltk_stopwords = stopwords.words("english")
    collection = Collection(
        name="cs276", stopwords_list=nltk_stopwords, lemmatizer=word_net_lemmatizer
    )
