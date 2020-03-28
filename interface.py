import time

from nltk.stem import WordNetLemmatizer
import click
import pyfiglet

from search_engine import SearchEngine


@click.command()
@click.option("-c", "--count", default=10, help="Number of results.", type=int)
@click.option(
    "-w",
    "--weighting_model",
    default="tw-idf",
    type=click.Choice(["tw-idf", "tf-idf", "okapi-bm25"], case_sensitive=False),
    help="Weighting Type.",
)
def interface(count, weighting_model):
    click.clear()
    click.secho("Loading search engine ...", fg="blue", bold=True)
    word_net_lemmatizer = WordNetLemmatizer()
    search_engine = SearchEngine(
        collection_name="cs276",
        stopwords_list=[],
        lemmatizer=word_net_lemmatizer,
        weighting_model=weighting_model,
    )
    click.clear()

    def search(query):
        """
        This function is the main interface for querying the search engine.
        """
        start_time = time.time()
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
        total_time = round((finished_time - start_time) * 1000, 2)
        click.secho(
            "Finished ! Total time: {}ms".format(total_time), fg="green", bold=True
        )

        for i, doc_id_query in enumerate(sorted_docs[:count]):
            document = search_engine.collection.documents[doc_id_query]
            click.secho(
                "{}.\t{}/{}".format(i, document.folder, document.url), bold=True
            )
            click.secho("\t{}\n".format(" ".join(document.key_words)), fg="red")

    while True:
        result = pyfiglet.figlet_format("Google 1998", font="big")
        click.secho(result, fg="red", bold=True)
        user_query = click.prompt(
            click.style("Please enter you query", fg="blue", bold=True), type=str
        )
        search(user_query)
        click.confirm("Do you want to continue?", abort=True)
        click.clear()


if __name__ == "__main__":
    interface()
