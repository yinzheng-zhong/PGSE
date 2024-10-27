#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Function to get the complement of a nucleotide sequence in-place
void get_complement(const char *seq, char *complement_seq, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        switch (seq[len - i - 1]) {
            case 'a': complement_seq[i] = 't'; break;
            case 't': complement_seq[i] = 'a'; break;
            case 'g': complement_seq[i] = 'c'; break;
            case 'c': complement_seq[i] = 'g'; break;
            default:  complement_seq[i] = 'n'; break;
        }
    }
    complement_seq[len] = '\0';
}

// Function to compare two strings lexicographically
int is_canonical(const char *seq, const char *complement_seq) {
    return strcmp(seq, complement_seq) <= 0;
}

// Optimized function to count overlapping occurrences of a substring in a string
int count_overlapping(const char *str, const char *sub, size_t sub_len) {
    int count = 0;
    const char *p = str;
    while ((p = strstr(p, sub)) != NULL) {
        count++;
        p++;  // Move one character forward for overlapping counting
    }
    return count;
}

// Main function to count segments in nodes
void count_segments(
    char **nodes, int num_nodes,
    char **segments, int num_segments,
    int *result_counts
) {
    // Initialize result_counts to zero
    memset(result_counts, 0, num_segments * sizeof(int));

    // Buffer for complement to avoid frequent malloc/free
    char complement_buffer[256];

    // For each node
    for (int n = 0; n < num_nodes; ++n) {
        char *node = nodes[n];
        size_t node_len = strlen(node);

        // For each segment
        for (int s = 0; s < num_segments; ++s) {
            char *segment = segments[s];
            size_t seg_len = strlen(segment);

            // Skip if segment is longer than node
            if (seg_len > node_len) continue;

            // Get complement of the segment once per segment
            get_complement(segment, complement_buffer, seg_len);

            // Check if segment is canonical and count occurrences
            if (is_canonical(segment, complement_buffer)) {
                int count = count_overlapping(node, segment, seg_len);

                // If segment is not palindromic, count occurrences of complement
                if (strcmp(segment, complement_buffer) != 0) {
                    count += count_overlapping(node, complement_buffer, seg_len);
                }

                // Update result
                result_counts[s] += count;
            }
        }
    }
}
