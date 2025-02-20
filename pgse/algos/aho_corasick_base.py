class AhoCorasickBase:
    def __init__(self):
        self.automaton = None
        pass

    def count_segments(self, nodes, segments):
        raise NotImplementedError
