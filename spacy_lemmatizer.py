import spacy


class SpacyLemmatizer:
    def __init__(self):
        print("Loading Spacy NLP French Model : fr_core_news_md")
        self.__nlp = spacy.load("fr_core_news_md")
        print("Finish Loading")

    def lemmatize(self, word):
        lemmatized_word = self.__nlp(word)[0]
        return lemmatized_word.lemma_


if __name__ == "__main__":
    spacy_french_lemmatizer = SpacyLemmatizer()
    sentence = "le chien est beau"
    print(spacy_french_lemmatizer.lemmatize(sentence))
