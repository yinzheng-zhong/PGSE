#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Function to get the complement of a nucleotide sequence
void get_complement(const char *seq, char *complement_seq) {
    size_t len = strlen(seq);
    for (size_t i = 0; i < len; ++i) {
        switch (seq[len - i - 1]) {
            case 'a':
                complement_seq[i] = 't';
                break;
            case 't':
                complement_seq[i] = 'a';
                break;
            case 'g':
                complement_seq[i] = 'c';
                break;
            case 'c':
                complement_seq[i] = 'g';
                break;
            default:
                complement_seq[i] = 'n';
                break;
        }
    }
    complement_seq[len] = '\0';
}

// Function to compare two strings lexicographically
int is_canonical(const char *seq, const char *complement_seq) {
    int cmp = strcmp(seq, complement_seq);
    if (cmp < 0) {
        return 1;  // seq is canonical
    } else if (cmp > 0) {
        return 0;  // complement_seq is canonical
    } else {
        return 1;  // Palindromic sequences are considered canonical
    }
}

// Function to count overlapping occurrences of a substring in a string
int count_overlapping(const char *str, const char *sub) {
    int count = 0;
    size_t sub_len = strlen(sub);
    if (sub_len == 0) return 0;

    const char *p = str;
    while ((p = strstr(p, sub)) != NULL) {
        count++;
        p++;  // Move one character forward to count overlapping
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
    for (int i = 0; i < num_segments; ++i) {
        result_counts[i] = 0;
    }

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

            // Get complement of the segment
            char *complement_segment = (char *)malloc((seg_len + 1) * sizeof(char));
            get_complement(segment, complement_segment);

            // Check if segment is canonical
            if (!is_canonical(segment, complement_segment)) {
                free(complement_segment);
                continue;
            }

            // Count occurrences of segment in node
            int count = count_overlapping(node, segment);

            // If segment is not palindromic, count occurrences of complement
            if (strcmp(segment, complement_segment) != 0) {
                count += count_overlapping(node, complement_segment);
            }

            // Update result
            result_counts[s] += count;

            free(complement_segment);
        }
    }
}
