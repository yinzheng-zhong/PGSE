# Function to check if a sequence is canonical
from pgse.algos.aho_corasick_base import AhoCorasickBase
from pgse.algos.utils import get_complement,  is_canonical


# Trie Node class for the Aho-Corasick automaton
class TrieNode:
    def __init__(self):
        self.children = {}  # Children nodes
        self.fail = None  # Failure link
        self.outputs = []  # Output patterns ending at this node


class AhoCorasickPy(AhoCorasickBase):
    def __init__(self):
        super().__init__()

    # Build the automaton
    def build_automaton(self, segments):
        # Preprocess segments to get canonical forms
        patterns = []  # Patterns to insert into the automaton
        pattern_indices = []  # Corresponding indices in the segments list

        for idx, segment in enumerate(segments):
            segment = segment.lower()
            complement = get_complement(segment)
            if is_canonical(segment, complement):
                patterns.append(segment)
                pattern_indices.append(idx)
                # If not palindromic, include complement
                if segment != complement:
                    patterns.append(complement)
                    pattern_indices.append(idx)
            else:
                # Skip non-canonical sequences
                continue

        root = TrieNode()
        # Insert patterns into the trie
        for idx, pattern in enumerate(patterns):
            node = root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.outputs.append(pattern_indices[idx])  # Store the original segment index

        # Build failure links
        from collections import deque
        queue = deque()
        # Set failure links of root's immediate children to root
        for child in root.children.values():
            child.fail = root
            queue.append(child)

        # BFS traversal to set failure links
        while queue:
            current_node = queue.popleft()
            for char, child_node in current_node.children.items():
                # Set the failure link for child_node
                fail_node = current_node.fail
                while fail_node and char not in fail_node.children:
                    fail_node = fail_node.fail
                child_node.fail = fail_node.children[char] if fail_node and char in fail_node.children else root
                child_node.outputs.extend(child_node.fail.outputs)  # Merge outputs
                queue.append(child_node)

        self.automaton = root
        return root

    # Function to search patterns in the text using the automaton
    def _search_automaton(self, root, text, result_counts):
        node = root
        for i in range(len(text)):
            char = text[i]
            while node != root and char not in node.children:
                node = node.fail
            if char in node.children:
                node = node.children[char]
            else:
                continue
            # If there are outputs, increment counts
            for pattern_idx in node.outputs:
                result_counts[pattern_idx] += 1  # Overlapping matches
        return result_counts

    # Main function to count segments in nodes using Aho-Corasick algorithm
    def count_segments(self, nodes, segments):
        # Build the automaton with canonical patterns
        root = self.build_automaton(segments)

        # Initialize result counts
        result_counts = [0] * len(segments)

        # Search each node's text
        for node_text in nodes:
            node_text = node_text.lower()
            self._search_automaton(root, node_text, result_counts)

        return result_counts