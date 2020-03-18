from collection import Collection
from query import Query
from helpers import merge_and_postings_list
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from time import time


class SearchEngine:
    def __init__(self, collection_name: str, stopwords_list, lemmatizer):
        self.collection = Collection(collection_name, stopwords_list, lemmatizer)
        self.stopwords = stopwords_list
        self.lemmatizer = lemmatizer

    def search(self, string_query: str):
        query = Query(string_query.lower(), self.stopwords, self.lemmatizer)
        posting_list = self.__get_posting_list(query)
        doc_scores = self.__get_score(posting_list, query)
        return doc_scores

    def __get_posting_list(self, query: Query):
        """

        :param query: Query object tokenize and processed
        :return: final_posting_list : ordered list of relevant documents for this query

        final_posting_list is obtained by merging the posting lists of each document
        """
        final_posting_list = []
        vocabulary = query.get_vocabulary()
        for token in vocabulary:
            if not final_posting_list:
                final_posting_list = self.collection.get_posting_list(token)
            else:
                final_posting_list = merge_and_postings_list(
                    final_posting_list, self.collection.get_posting_list(token)
                )
        return final_posting_list

    def __get_score(self, posting_list, query: Query):
        """

        :param posting_list: the list of documents which contain the terms of the query
        :param query: tokenized and processed query
        :return: a dictionary of keys : documents of the posting_list, values : their score for this query

        In this function we first calculate each tf score for the terms of the query.
        """
        query_vector = [0 for _ in range(len(self.collection.term_index))]
        for token in query.get_vocabulary():
            try:
                id_term = self.collection.term_index[token]
            except KeyError:
                continue
            tf_idf = query.get_tf(token) * self.collection.get_idf(token)
            query_vector[id_term] = tf_idf
        query_vector = sparse.csr_matrix(query_vector)
        doc_score = {}
        for doc_id in posting_list:
            doc_vector = [0 for _ in range(len(self.collection.term_index))]
            for token in self.collection.documents[doc_id].get_vocabulary():
                try:
                    id_token = self.collection.term_index[token]
                    tw_idf = self.collection.get_tw_idf(
                        target_term=token, target_doc_id=doc_id, b=0.003
                    )
                    doc_vector[id_token] = tw_idf
                except KeyError as e:
                    continue
            doc_vector = sparse.csr_matrix(doc_vector)
            score = cosine_similarity(query_vector, doc_vector)
            doc_score[doc_id] = score[0][0]
        return doc_score


if __name__ == "__main__":
    nltk_stopwords = stopwords.words("english")
    word_net_lemmatizer = WordNetLemmatizer()
    search_engine = SearchEngine(
        collection_name="cs276",
        stopwords_list=nltk_stopwords,
        lemmatizer=word_net_lemmatizer,
    )

    for i in range(3, 9):
        start = time()
        with (open("dev_queries/query.{}".format(str(i)), "r")) as query_file:
            query_content = next(query_file).rstrip("\n")
        print(query_content)
        doc_scores = search_engine.search(query_content)
        sorted_docs = [
            k
            for k, v in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]
        print("Process {} duration : {} seconds".format(str(i), time() - start))
        with open("dev_predictions/query{}.out".format(str(i)), "w") as result_file:
            for doc_id in sorted_docs:
                document = search_engine.collection.documents[doc_id]
                line = "{}/{} {}".format(
                    document.folder, document.url, doc_scores[doc_id]
                )
                result_file.write(line + "\n")
        print("Query{} duration : {} seconds".format(str(i), time() - start))
