#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALPHABET_SIZE 4  // For DNA sequences: 'a', 'c', 'g', 't'

// Mapping nucleotides to indices
int char_to_index(char c) {
    switch (c) {
        case 'a': return 0;
        case 'c': return 1;
        case 'g': return 2;
        case 't': return 3;
        default: return -1;
    }
}

// Node structure for the trie
typedef struct TrieNode {
    struct TrieNode *children[ALPHABET_SIZE];
    struct TrieNode *fail;  // Failure link
    int *output;            // Output function: array of segment indices
    int output_size;
    int output_capacity;
} TrieNode;

// Function to create a new trie node
TrieNode* create_trie_node() {
    TrieNode *node = (TrieNode *)malloc(sizeof(TrieNode));
    for (int i = 0; i < ALPHABET_SIZE; ++i) node->children[i] = NULL;
    node->fail = NULL;
    node->output = NULL;
    node->output_size = 0;
    node->output_capacity = 0;
    return node;
}

// Function to insert a pattern into the trie
void insert_pattern(TrieNode *root, const char *pattern, int pattern_index) {
    TrieNode *node = root;
    int idx;
    for (int i = 0; pattern[i]; ++i) {
        idx = char_to_index(pattern[i]);
        if (idx == -1) continue;  // Skip invalid characters
        if (node->children[idx] == NULL) {
            node->children[idx] = create_trie_node();
        }
        node = node->children[idx];
    }
    // Append pattern_index to output list
    if (node->output_capacity == 0) {
        node->output_capacity = 2;
        node->output = (int *)malloc(node->output_capacity * sizeof(int));
    } else if (node->output_size == node->output_capacity) {
        node->output_capacity *= 2;
        node->output = (int *)realloc(node->output, node->output_capacity * sizeof(int));
    }
    node->output[node->output_size++] = pattern_index;
}

// Function to build failure links
void build_failure_links(TrieNode *root) {
    // Initialize dynamic queue
    int queue_capacity = 1000;
    TrieNode **queue = (TrieNode **)malloc(queue_capacity * sizeof(TrieNode *));
    int front = 0, rear = 0;

    // Initialize failure links of root's immediate children to root
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        if (root->children[i]) {
            root->children[i]->fail = root;
            // Ensure queue capacity
            if (rear >= queue_capacity) {
                queue_capacity *= 2;
                queue = (TrieNode **)realloc(queue, queue_capacity * sizeof(TrieNode *));
            }
            queue[rear++] = root->children[i];
        }
    }

    // BFS traversal to set failure links
    while (front < rear) {
        TrieNode *current = queue[front++];
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            TrieNode *child = current->children[i];
            if (child) {
                TrieNode *fail_node = current->fail;
                while (fail_node && !fail_node->children[i]) {
                    fail_node = fail_node->fail;
                }
                if (fail_node) {
                    child->fail = fail_node->children[i];
                } else {
                    child->fail = root;
                }
                // Merge output functions
                if (child->fail->output_size > 0) {
                    int total_output = child->output_size + child->fail->output_size;
                    if (total_output > child->output_capacity) {
                        child->output_capacity = total_output;
                        child->output = (int *)realloc(child->output, child->output_capacity * sizeof(int));
                    }
                    memcpy(child->output + child->output_size, child->fail->output, child->fail->output_size * sizeof(int));
                    child->output_size = total_output;
                }
                // Ensure queue capacity
                if (rear >= queue_capacity) {
                    queue_capacity *= 2;
                    queue = (TrieNode **)realloc(queue, queue_capacity * sizeof(TrieNode *));
                }
                queue[rear++] = child;
            }
        }
    }
    free(queue);
}

// Function to get the complement of a nucleotide sequence
void get_complement(const char *seq, char *complement_seq) {
    size_t len = strlen(seq);
    for (size_t i = 0; i < len; ++i) {
        switch (seq[len - i - 1]) {
            case 'a': complement_seq[i] = 't'; break;
            case 't': complement_seq[i] = 'a'; break;
            case 'g': complement_seq[i] = 'c'; break;
            case 'c': complement_seq[i] = 'g'; break;
            default: complement_seq[i] = 'n'; break;
        }
    }
    complement_seq[len] = '\0';
}

// Function to compare two strings lexicographically
int is_canonical(const char *seq, const char *complement_seq) {
    return strcmp(seq, complement_seq) <= 0;
}

// Function to process the text using the Aho-Corasick automaton
void search_in_text(TrieNode *root, const char *text, int *result_counts) {
    TrieNode *node = root;
    int idx;
    for (int i = 0; text[i]; ++i) {
        idx = char_to_index(text[i]);
        if (idx == -1) {
            node = root;  // Reset to root on invalid character
            continue;
        }
        while (node != root && node->children[idx] == NULL) {
            node = node->fail;
        }
        if (node->children[idx]) {
            node = node->children[idx];
        }
        // Process outputs
        if (node->output_size > 0) {
            for (int j = 0; j < node->output_size; ++j) {
                result_counts[node->output[j]]++;
            }
        }
    }
}

// Function to free the trie
void free_trie(TrieNode *root) {
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        if (root->children[i]) {
            free_trie(root->children[i]);
        }
    }
    if (root->output) free(root->output);
    free(root);
}

// Function to build the Aho-Corasick automaton
TrieNode* build_automaton(char **segments, int num_segments) {
    // Precompute complements and canonical status
    char **complements = (char **)malloc(num_segments * sizeof(char *));
    int *canonical = (int *)malloc(num_segments * sizeof(int));

    for (int s = 0; s < num_segments; ++s) {
        size_t len = strlen(segments[s]);
        complements[s] = (char *)malloc((len + 1) * sizeof(char));
        get_complement(segments[s], complements[s]);
        canonical[s] = is_canonical(segments[s], complements[s]);
    }

    // Build the Aho-Corasick automaton
    TrieNode *root = create_trie_node();

    for (int s = 0; s < num_segments; ++s) {
        if (!canonical[s]) continue;  // Skip non-canonical segments

        // Insert the segment
        insert_pattern(root, segments[s], s);

        // If segment is not palindromic, insert its complement
        if (strcmp(segments[s], complements[s]) != 0) {
            insert_pattern(root, complements[s], s);
        }
    }

    // Build failure links
    build_failure_links(root);

    // Free allocated memory for complements and canonical
    for (int s = 0; s < num_segments; ++s) {
        free(complements[s]);
    }
    free(complements);
    free(canonical);

    return root;
}

// Function to free the automaton
void free_automaton(TrieNode* root) {
    free_trie(root);
}

// Function to process the text using the automaton
void process_nodes(TrieNode* root, char **nodes, int num_nodes, int *result_counts, int num_segments) {
    // Initialize result_counts to zero
    for (int i = 0; i < num_segments; ++i) {
        result_counts[i] = 0;
    }

    // Process each node
    for (int n = 0; n < num_nodes; ++n) {
        search_in_text(root, nodes[n], result_counts);
    }
}

// Keep the original count_segments function for compatibility

void count_segments(
    char **nodes, int num_nodes,
    char **segments, int num_segments,
    int *result_counts
) {
    // Build the automaton
    TrieNode *root = build_automaton(segments, num_segments);

    // Process the nodes
    process_nodes(root, nodes, num_nodes, result_counts, num_segments);

    // Free the automaton
    free_automaton(root);
}

