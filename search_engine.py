from collection import Collection
from os import listdir
from query import Query
from helpers import merge_and_postings_list, merge_or_postings_list
from nltk.corpus import stopwords
from spacy_lemmatizer import SpacyLemmatizer
from nltk.stem import WordNetLemmatizer
from time import time
from math import sqrt


class SearchEngine:
    def __init__(self, collection_name: str, stopwords_list, lemmatizer):
        self.collection = Collection(collection_name, stopwords_list, lemmatizer)
        self.stopwords = stopwords_list
        self.lemmatizer = lemmatizer

    def search(self, string_query: str):
        query = Query(string_query.lower(), self.stopwords, self.lemmatizer)
        posting_list = self.__get_posting_list(query)
        doc_scores = self.__get_scores(posting_list, query)
        return doc_scores

    def __get_posting_list(self, query: Query):
        final_posting_list = []
        vocabulary = query.get_vocabulary()
        for token in vocabulary:
            if not final_posting_list:
                final_posting_list = self.collection.get_posting_list(token)
            else:
                final_posting_list = merge_or_postings_list(
                    final_posting_list, self.collection.get_posting_list(token)
                )
        return final_posting_list

    def __get_scores(self, posting_list, query: Query):
        query_tf_idf = {}
        norm_query_vector = 0
        query_vocabulary = query.get_vocabulary()
        for token in query_vocabulary:
            tf_idf = query.get_tf(token) * self.collection.get_idf(token)
            query_tf_idf[token] = tf_idf
            norm_query_vector += tf_idf ** 2
        norm_query_vector = sqrt(norm_query_vector)
        doc_scores = {}
        for doc_id in posting_list:
            score = 0
            for token in query_vocabulary:
                tw_idf = self.collection.get_tw_idf(
                    target_term=token, target_doc_id=doc_id, b=0.003
                )
                score += query_tf_idf[token] * tw_idf
            score /= self.collection.documents_norms[doc_id] * norm_query_vector
            doc_scores[doc_id] = score
        return doc_scores


if __name__ == "__main__":
    spacy_french_lemmatizer = SpacyLemmatizer()
    nltk_stopwords = stopwords.words("french")
    search_engine = SearchEngine(
        collection_name="fquad",
        stopwords_list=stopwords.words("french"),
        lemmatizer=spacy_french_lemmatizer,
    )
    nb_queries = len(listdir("fquad_queries"))
    """
    nltk_stopwords = stopwords.words("english")
    word_net_lemmatizer = WordNetLemmatizer()
    search_engine = SearchEngine(
        collection_name="cs276",
        stopwords_list=nltk_stopwords,
        lemmatizer=word_net_lemmatizer,
    )
    """
    for i in range(19113, nb_queries):
        start = time()
        with (open("fquad_queries/query.{}".format(str(i)), "r")) as query_file:
            query_content = next(query_file).rstrip("\n")
        print(query_content)
        doc_scores_query = search_engine.search(query_content)
        sorted_docs = [
            k
            for k, v in sorted(
                doc_scores_query.items(), key=lambda item: item[1], reverse=True
            )
        ]
        with open("fquad_predictions/out.{}".format(str(i)), "w") as result_file:
            for doc_id_query in sorted_docs:
                document = search_engine.collection.documents[doc_id_query]
                line = "{}/{} {}".format(
                    document.folder, document.url, doc_scores_query[doc_id_query]
                )
                result_file.write(line + "\n")
        print("Query{} duration : {} seconds".format(str(i), time() - start))
