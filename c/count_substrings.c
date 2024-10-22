// count_substrings.c
#include <string.h>

// Function to count overlapping occurrences of `sub` in `str`
int count_substrings(const char *str, const char *sub) {
    int count = 0;
    int sub_len = strlen(sub);

    if (sub_len == 0) {
        return 0; // No meaningful substring
    }

    for (const char *p = str; (p = strstr(p, sub)); ++p) {
        count++;
    }

    return count;
}
