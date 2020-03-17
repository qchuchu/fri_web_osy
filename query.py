class Query:
    def __init__(self, content, stopwords_list, lemmatizer):
        self.content = content
        self.stopwords = stopwords_list
        self.lemmatizer = lemmatizer
        self.tokens = content.split(" ")
        self.__length = len(self.tokens)
        self.term_frequencies = {}
        self.__process_query(self.stopwords, self.lemmatizer)

    def __remove_stopwords(self, stopwords_list):
        self.tokens = [token for token in self.tokens if token not in stopwords_list]
        self.__length = len(self.tokens)

    def __lemmatize(self, lemmatizer):
        self.tokens = [lemmatizer.lemmatize(token) for token in self.tokens]

    def __get_term_frequencies(self):
        for token in self.tokens:
            if token in self.term_frequencies:
                self.term_frequencies[token] += 1
            else:
                self.term_frequencies[token] = 1

    def __process_query(self, stopwords_list, lemmatizer):
        self.__remove_stopwords(stopwords_list)
        self.__lemmatize(lemmatizer)
        self.__get_term_frequencies()

    def get_tf(self, target_term):
        try:
            tf = self.term_frequencies[target_term]
        except KeyError:
            return 0
        return tf

    def get_vocabulary(self):
        return list(self.term_frequencies.keys())
