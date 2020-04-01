import click
from time import time
from os.path import isfile
from nltk.stem import WordNetLemmatizer

from utils.helpers import merge_and_postings_list
from models.search_engine import SearchEngine


@click.command()
@click.option(
    "-w",
    "--weighting_model",
    default="tw-idf",
    type=click.Choice(["tw-idf", "tf-idf", "okapi-bm25"], case_sensitive=False),
    help="Weighting Type.",
)
def show_accuracy(weighting_model):
    """
    This function compares the results from the dev predictions and dev output to evaluate the
    performance of a model
    :return:
    """
    if not isfile("dev_resources/predictions/1_{}.out".format(weighting_model)):
        compute_dev_predictions(weighting_model)
    for i in range(1, 9):
        dev_output = []
        search_engine_output = []
        with open("dev_resources/output/{}.out".format(i), "r") as file:
            reader = file.readlines()
            for line in reader:
                dev_output.append(line.rstrip("\n"))
        with open(
            "dev_resources/predictions/{}_{}.out".format(i, weighting_model), "r"
        ) as file:
            reader = file.readlines()
            for line in reader:
                parsed_line = line.rstrip("\n").split(" ")
                search_engine_output.append(parsed_line[0])
        with open("dev_resources/queries/query.{}".format(i), "r") as file:
            query_content = next(file).rstrip("\n")
        if len(search_engine_output) == 0:
            print("Search Engine has failed on Query {}".format(i))
        else:
            search_engine_output = search_engine_output[: len(dev_output)]

            posting_term1 = sorted(dev_output)
            posting_term2 = sorted(search_engine_output)

            final_list = merge_and_postings_list(posting_term1, posting_term2)
            score = len(final_list) / len(dev_output)

            print(
                'For Query {} : "{}" the accuracy score is {}%'.format(
                    i, query_content, "{0:.2f}".format(score * 100)
                )
            )


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
        with (open("dev_resources/queries/query.{}".format(str(i)), "r")) as query_file:
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
            "dev_resources/predictions/{}_{}.out".format(str(i), weighting_model), "w"
        ) as result_file:
            for doc_id_query in sorted_docs:
                document = search_engine.collection.documents[doc_id_query]
                line = "{}/{} {}".format(
                    document.folder, document.url, doc_scores_query[doc_id_query]
                )
                result_file.write(line + "\n")
        print("Query{} duration : {} seconds".format(str(i), time() - start))


if __name__ == "__main__":
    show_accuracy()
