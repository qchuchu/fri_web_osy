from json import load
from os import mkdir
from nltk.tokenize import RegexpTokenizer


def extract_data_from_fquad():
    with open("fquad_train.json") as json_file:
        json_data = load(json_file)
        articles = json_data["data"]
        titles = list(map(lambda x: x["title"], articles))
        questions = []
        y_article = []
        contexts = []
        y_context = []
        # This will give the mapping of a context --> article
        context_article_mapping = []
        # We get into an article
        for index_article, article in enumerate(articles):
            # We get into the paragraphs
            paragraphs = article["paragraphs"]
            for paragraph in paragraphs:
                qas = paragraph["qas"]
                context = paragraph["context"]
                questions_paragraph = list(map(lambda x: x["question"], qas))
                nb_questions = len(questions_paragraph)
                id_context = len(contexts)
                # We add all the data
                contexts.append(context)
                context_article_mapping.append(index_article)
                questions.extend(questions_paragraph)
                y_context.extend([id_context for _ in range(nb_questions)])
                y_article.extend([index_article for _ in range(nb_questions)])
    """
    mkdir('data/fquad')
    for i in range(len(articles)):
        mkdir('data/fquad/{}'.format(i))
    tokenizer = RegexpTokenizer(r'\w+')
    for j in range(len(contexts)):
        context_article_id = context_article_mapping[j]
        with open('data/fquad/{}/context_{}'.format(context_article_id, j), 'w') as file:
            tokenized_context = tokenizer.tokenize(contexts[j])
            file.write(" ".join(tokenized_context).lower())
    """
    mkdir("fquad_output")
    mkdir("fquad_queries")
    tokenizer = RegexpTokenizer(r"\w+")
    for index, question in enumerate(questions):
        with open("fquad_queries/query.{}".format(index), "w") as file:
            tokenized_question = tokenizer.tokenize(question)
            file.write(" ".join(tokenized_question).lower())
        with open("fquad_output/out.{}".format(index), "w") as file:
            output = "{}/context_{}".format(y_article[index], y_context[index])
            file.write(output)
    return titles, questions, contexts, y_article, y_context, context_article_mapping


if __name__ == "__main__":
    extract_data_from_fquad()
