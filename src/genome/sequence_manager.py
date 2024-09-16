class SequenceManager:
    def __init__(self):
        self.train_sequences = []  # Use a managed list
        self.test_sequences = []   # Use a managed list

    def add_train_sequence(self, sequence):
        self.train_sequences.append(sequence)

    def add_train_sequences(self, sequences):
        self.train_sequences.extend(sequences)

    def add_test_sequence(self, sequence):
        self.test_sequences.append(sequence)

    def add_test_sequences(self, sequences):
        self.test_sequences.extend(sequences)

    def get_train_sequence(self, index):
        return self.train_sequences[index]

    def get_test_sequence(self, index):
        return self.test_sequences[index]

    def clear(self):
        self.train_sequences.clear()
        self.test_sequences.clear()
