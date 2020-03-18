from os import listdir


def print_accuracy(k: int):
    """
    This function compares the results from the dev predictions and dev output to evaluate the
    performance of a model
    :return:
    """
    nb_queries = len(listdir("fquad_queries"))
    context_score = 0
    article_score = 0
    for i in range(nb_queries):
        dev_output = []
        search_engine_output = []
        with open("fquad_output/out.{}".format(i), "r") as file:
            reader = file.readlines()
            for line in reader:
                dev_output.append(line.rstrip("\n"))
        with open("fquad_predictions/out.{}".format(i), "r") as file:
            reader = file.readlines()
            for line in reader:
                parsed_line = line.rstrip("\n").split(" ")
                search_engine_output.append(parsed_line[0])
        if len(search_engine_output) > 0:
            for j in range(min(k, len(search_engine_output))):
                if search_engine_output[j] == dev_output[0]:
                    context_score += 1
                    break
            for p in range(min(k, len(search_engine_output))):
                article_pred, context_pred = search_engine_output[p].split("/")
                article_out, context_out = dev_output[0].split("/")
                if article_out == article_pred:
                    article_score += 1
                    break

    context_score /= nb_queries
    article_score /= nb_queries
    print(
        "The Context Score Accuracy with Top-{} precision is {}".format(
            k, "{0:.2f}".format(context_score * 100)
        )
    )
    print(
        "The Article Score Accuracy with Top-{} precision is {}".format(
            k, "{0:.2f}".format(article_score * 100)
        )
    )
    return context_score, article_score


if __name__ == "__main__":
    print(len(listdir("fquad_queries")))
    with open("fquad_retrieval_or_policy_tw_idf.txt", "w") as result_file:
        for k in range(1, 26):
            context_score, article_score = print_accuracy(k)
            result_file.write("{} {} {}\n".format(k, context_score, article_score))
