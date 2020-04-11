
class SortedList(list):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def insert(self, doc_score, doc_id):
        i = 0

        if len(self) < self.length:
            for value in self:
                if value[0] > doc_score:
                    super().insert(i, (doc_score, doc_id))
                    return
                i += 1
            super().insert(i, (doc_score, doc_id))

        elif doc_score > self.__getitem__(0)[0]:
            super().pop(0)
            for value in self:
                if value[0] > doc_score:
                    super().insert(i, (doc_score, doc_id))
                    return
                i += 1
            super().insert(i, (doc_score, doc_id))

