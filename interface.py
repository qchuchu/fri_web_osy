import time

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import click

from search_engine import SearchEngine


@click.command()
@click.option(
    "--query", required=True, prompt="Search query", help="Your search query", type=str
)
@click.option("--count", default=10, help="Number of results.", type=int)
def search(query, count):
    """
    This function is the main interface for querying the search engine.
    """
    start_time = time.time()

    click.secho("Loading search engine ...", fg="blue", bold=True)
    nltk_stopwords = stopwords.words("english")
    word_net_lemmatizer = WordNetLemmatizer()
    search_engine = SearchEngine(
        collection_name="cs276",
        stopwords_list=nltk_stopwords,
        lemmatizer=word_net_lemmatizer,
    )

    click.secho("Searching query ...", fg="blue", bold=True)
    doc_scores_query = search_engine.search(query)

    click.secho("Sorting results ...", fg="blue", bold=True)
    sorted_docs = [
        k
        for k, v in sorted(
            doc_scores_query.items(), key=lambda item: item[1], reverse=True
        )
    ]

    finished_time = time.time()
    total_time = round(finished_time - start_time, 2)
    click.secho("Finished ! Total time: {}s".format(total_time), fg="green", bold=True)

    for i, doc_id_query in enumerate(sorted_docs[:count]):
        document = search_engine.collection.documents[doc_id_query]
        click.secho("{}.\t{}/{}".format(i, document.folder, document.url), bold=True)
        click.secho("\t{}\n".format(" ".join(document.key_words)), fg="red")


if __name__ == "__main__":
    search()
