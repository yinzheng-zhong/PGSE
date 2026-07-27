#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTE_VALUES 256
#define DNA_ALPHABET_SIZE 4  // For DNA sequences: 'a', 'c', 'g', 't'

// Description of the alphabet the automaton runs on. Built by the caller so the
// same code can handle DNA, protein, or arbitrary text.
typedef struct Alphabet {
    const int *char_index;    // BYTE_VALUES entries, -1 for characters outside the alphabet
    int size;                 // Number of distinct characters
    const char *complement;   // BYTE_VALUES entries, or NULL to disable canonicalisation
} Alphabet;

// Node structure for the trie
typedef struct TrieNode {
    struct TrieNode **children;  // One slot per alphabet character
    struct TrieNode *fail;       // Failure link
    int *output;                 // Output function: array of segment indices
    int output_size;
    int output_capacity;
} TrieNode;

// Function to create a new trie node
TrieNode* create_trie_node(int alphabet_size) {
    TrieNode *node = (TrieNode *)malloc(sizeof(TrieNode));
    node->children = (TrieNode **)calloc(alphabet_size, sizeof(TrieNode *));
    node->fail = NULL;
    node->output = NULL;
    node->output_size = 0;
    node->output_capacity = 0;
    return node;
}

// Function to insert a pattern into the trie
void insert_pattern(TrieNode *root, const Alphabet *alphabet, const char *pattern, int pattern_index) {
    TrieNode *node = root;
    int idx;

    // A pattern that is empty, or that contains a character the alphabet does not
    // have, can never match. Dropping it whole is important: skipping just the
    // offending characters would shorten the pattern, and a pattern shortened to
    // nothing would hang its output off the root and match at every position.
    if (pattern[0] == '\0') return;
    for (int i = 0; pattern[i]; ++i) {
        if (alphabet->char_index[(unsigned char)pattern[i]] == -1) return;
    }

    for (int i = 0; pattern[i]; ++i) {
        idx = alphabet->char_index[(unsigned char)pattern[i]];
        if (node->children[idx] == NULL) {
            node->children[idx] = create_trie_node(alphabet->size);
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
void build_failure_links(TrieNode *root, int alphabet_size) {
    // Initialize dynamic queue
    int queue_capacity = 1000;
    TrieNode **queue = (TrieNode **)malloc(queue_capacity * sizeof(TrieNode *));
    int front = 0, rear = 0;

    // Initialize failure links of root's immediate children to root
    for (int i = 0; i < alphabet_size; ++i) {
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
        for (int i = 0; i < alphabet_size; ++i) {
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

// Function to get the complement of a sequence. Alphabets without a complement copy
// the sequence unchanged, which makes canonicalisation a no-op.
void get_complement(const Alphabet *alphabet, const char *seq, char *complement_seq) {
    size_t len = strlen(seq);
    if (alphabet->complement == NULL) {
        memcpy(complement_seq, seq, len);
    } else {
        for (size_t i = 0; i < len; ++i) {
            complement_seq[i] = alphabet->complement[(unsigned char)seq[len - i - 1]];
        }
    }
    complement_seq[len] = '\0';
}

// Function to compare two strings lexicographically
int is_canonical(const char *seq, const char *complement_seq) {
    return strcmp(seq, complement_seq) <= 0;
}

// Function to process the text using the Aho-Corasick automaton
void search_in_text(TrieNode *root, const Alphabet *alphabet, const char *text, int *result_counts) {
    TrieNode *node = root;
    int idx;
    for (int i = 0; text[i]; ++i) {
        idx = alphabet->char_index[(unsigned char)text[i]];
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
void free_trie(TrieNode *root, int alphabet_size) {
    for (int i = 0; i < alphabet_size; ++i) {
        if (root->children[i]) {
            free_trie(root->children[i], alphabet_size);
        }
    }
    if (root->output) free(root->output);
    free(root->children);
    free(root);
}

// Main function to count segments in nodes using Aho-Corasick algorithm.
// char_index maps each byte to its position in the alphabet (-1 when outside it) and
// complement_table maps each byte to its complement, or is NULL when the alphabet has
// no complement and segments must therefore be counted as given.
void count_segments_ex(
    char **nodes, int num_nodes,
    char **segments, int num_segments,
    int *result_counts,
    const int *char_index, int alphabet_size,
    const char *complement_table
) {
    Alphabet alphabet = { char_index, alphabet_size, complement_table };

    // Initialize result_counts to zero
    for (int i = 0; i < num_segments; ++i) {
        result_counts[i] = 0;
    }

    // Precompute complements and canonical status
    char **complements = (char **)malloc(num_segments * sizeof(char *));
    int *canonical = (int *)malloc(num_segments * sizeof(int));

    for (int s = 0; s < num_segments; ++s) {
        size_t len = strlen(segments[s]);
        complements[s] = (char *)malloc((len + 1) * sizeof(char));
        get_complement(&alphabet, segments[s], complements[s]);
        canonical[s] = is_canonical(segments[s], complements[s]);
    }

    // Build the Aho-Corasick automaton
    TrieNode *root = create_trie_node(alphabet_size);

    for (int s = 0; s < num_segments; ++s) {
        if (!canonical[s]) continue;  // Skip non-canonical segments

        // Insert the segment
        insert_pattern(root, &alphabet, segments[s], s);

        // If segment is not palindromic, insert its complement
        if (strcmp(segments[s], complements[s]) != 0) {
            insert_pattern(root, &alphabet, complements[s], s);
        }
    }

    // Build failure links
    build_failure_links(root, alphabet_size);

    // Process each node
    for (int n = 0; n < num_nodes; ++n) {
        search_in_text(root, &alphabet, nodes[n], result_counts);
    }

    // Free allocated memory
    for (int s = 0; s < num_segments; ++s) {
        free(complements[s]);
    }
    free(complements);
    free(canonical);
    free_trie(root, alphabet_size);
}

// Backwards-compatible entry point: lowercase DNA with reverse complementing.
// 'n' marks a read error and is a character in its own right, so that segments
// carrying read errors can be matched.
void count_segments(
    char **nodes, int num_nodes,
    char **segments, int num_segments,
    int *result_counts
) {
    static const char dna[DNA_ALPHABET_SIZE + 1] = {'a', 'c', 'g', 't', 'n'};
    static const char dna_complement[DNA_ALPHABET_SIZE + 1] = {'t', 'g', 'c', 'a', 'n'};

    int char_index[BYTE_VALUES];
    char complement_table[BYTE_VALUES];

    for (int i = 0; i < BYTE_VALUES; ++i) {
        char_index[i] = -1;
        complement_table[i] = (char)i;
    }
    for (int i = 0; i < DNA_ALPHABET_SIZE + 1; ++i) {
        char_index[(unsigned char)dna[i]] = i;
        char_index[(unsigned char)(dna[i] - 'a' + 'A')] = i;
        complement_table[(unsigned char)dna[i]] = dna_complement[i];
        complement_table[(unsigned char)(dna[i] - 'a' + 'A')] = (char)(dna_complement[i] - 'a' + 'A');
    }

    count_segments_ex(
        nodes, num_nodes,
        segments, num_segments,
        result_counts,
        char_index, DNA_ALPHABET_SIZE + 1,
        complement_table
    );
}
