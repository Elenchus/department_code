class MockWv():
    vocab = {}

class MockW2V():
    def __init__(self, vocab = {}):
        self.wv = MockWv()
        self.wv.vocab = vocab

    def __getitem__(self, key):
        if len(self.wv.vocab) == 0:
            raise AttributeError("vocab has not been created")
        else:
            return self.wv.vocab[key]
