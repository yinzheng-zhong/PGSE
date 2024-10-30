class AhoCorasickBase:
    def __init__(self):
        self.automaton = None
        pass

    def count_segments(self, nodes, segments):
        raise NotImplementedError

    def build_automaton(self, segments):
        raise NotImplementedError

    def process_nodes(self, nodes, num_segments):
        raise NotImplementedError

