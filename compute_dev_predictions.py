from time import time

from nltk.stem import WordNetLemmatizer

from search_engine import SearchEngine


def compute_dev_predictions(weighting_model: str = "tw-idf"):
    word_net_lemmatizer = WordNetLemmatizer()
    search_engine = SearchEngine(
        collection_name="cs276",
        stopwords_list=[],
        lemmatizer=word_net_lemmatizer,
        weighting_model=weighting_model,
    )

    for i in range(1, 9):
        start = time()
        with (open("dev_queries/query.{}".format(str(i)), "r")) as query_file:
            query_content = next(query_file).rstrip("\n")
        print(query_content)
        doc_scores_query = search_engine.search(query_content)
        sorted_docs = [
            k
            for k, v in sorted(
                doc_scores_query.items(), key=lambda item: item[1], reverse=True
            )
        ]
        with open(
            "dev_predictions/{}_{}.out".format(str(i), weighting_model), "w"
        ) as result_file:
            for doc_id_query in sorted_docs:
                document = search_engine.collection.documents[doc_id_query]
                line = "{}/{} {}".format(
                    document.folder, document.url, doc_scores_query[doc_id_query]
                )
                result_file.write(line + "\n")
        print("Query{} duration : {} seconds".format(str(i), time() - start))


if __name__ == "__main__":
    compute_dev_predictions()
