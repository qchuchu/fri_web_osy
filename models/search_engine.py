from math import sqrt
from time import time

from nltk.stem import WordNetLemmatizer
import click
import numpy as np

from models.collection import Collection
from models.query import Query
from utils.helpers import merge_and_postings_list
from models.SortedList import SortedList


class SearchEngine:
    def __init__(
        self,
        collection_name: str,
        stopwords_list,
        lemmatizer,
        weighting_model: str = "tw-idf",
    ):
        self.collection = Collection(
            collection_name, stopwords_list, lemmatizer, weighting_model
        )
        self.weighting_model = weighting_model
        self.stopwords = stopwords_list
        self.lemmatizer = lemmatizer

    def search(self, string_query: str):
        query = Query(string_query.lower(), self.stopwords, self.lemmatizer)
        posting_list = self.__get_posting_list(query)
        # doc_scores = self.__get_scores(posting_list, query)
        doc_scores = self.__get_scores_list_top_k(query, 100)

        return doc_scores

    def __get_posting_list(self, query: Query):
        final_posting_list = []
        vocabulary = query.get_vocabulary()
        for token in vocabulary:
            start_time = time()
            if not final_posting_list:
                final_posting_list = self.collection.get_posting_list(token)
                posting_list_time = round(time() - start_time, 2)
                click.secho(
                    "[Search Engine] Token: {} | Posting list: {} items | Time: {}s".format(
                        token, len(final_posting_list), posting_list_time
                    ),
                    fg="bright_blue",
                )
            else:
                posting_list = self.collection.get_posting_list(token)
                posting_list_time = round(time() - start_time, 2)
                click.secho(
                    "[Search Engine] Token: {} | Posting list: {} items | Time: {}s | ".format(
                        token, len(posting_list), posting_list_time
                    ),
                    fg="bright_blue",
                    nl=False,
                )
                click.secho("Merge posting list needed", fg="red", bold=True, nl=False)
                start_time = time()
                final_posting_list = merge_and_postings_list(
                    final_posting_list, posting_list
                )
                merge_time = round(time() - start_time, 2)
                click.secho(
                    " | Merge time: {}s | Final : {} terms".format(
                        merge_time, len(final_posting_list)
                    ),
                    fg="red",
                    bold=True,
                )
        return final_posting_list


    def __get_scores_list_top_k(self, query: Query, k: int):
        # pre computing
        scores = np.zeros(self.collection.nb_docs)
        vocabulary = query.get_vocabulary()
        query_tf_idf = {}
        norm_query_vector = 0
        best_values = SortedList(k * 10)
        for token in vocabulary:
            tf_idf = query.get_tf(token) * self.collection.get_idf(token)
            query_tf_idf[token] = tf_idf
            norm_query_vector += tf_idf ** 2
        norm_query_vector = sqrt(norm_query_vector)
        doc_scores = {}

        for token in vocabulary:
            best_values = self.collection.make_list_and_score(token, scores, best_values, query_tf_idf[token])
        for tup in best_values:
            doc_scores[tup[1]] = tup[0] / norm_query_vector
        return doc_scores


    def __get_scores(self, posting_list, query: Query):
        click.secho("[Search Engine] Computing search scores ...", fg="bright_blue")
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
                if self.weighting_model == "tw-idf":
                    weight = self.collection.get_tw_idf(
                        target_term=token, target_doc_id=doc_id, b=0.003
                    )
                elif self.weighting_model == "tf-idf":
                    weight = self.collection.get_piv_plus(
                        target_term=token, target_doc_id=doc_id, b=0.2
                    )
                else:
                    weight = self.collection.get_bm25_plus(
                        target_term=token, target_doc_id=doc_id, b=0.75, k1=1.2
                    )
                score += query_tf_idf[token] * weight
            score /= self.collection.documents_norms[doc_id] * norm_query_vector
            doc_scores[doc_id] = score
        return doc_scores


if __name__ == "__main__":
    word_net_lemmatizer = WordNetLemmatizer()
    search_engine = SearchEngine(
        collection_name="cs276",
        stopwords_list=[],
        lemmatizer=word_net_lemmatizer,
    )
    doc_scores = search_engine.search("stanford class")
