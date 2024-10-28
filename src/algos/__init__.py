from src.algos.aho_corasick_py import count_segments as count_segments_py
from src.algos.aho_corasick_c import count_segments as _count_segments_c, load_lib
from src.log import logger

try:
    aho_corasick_c_lib = load_lib()
except FileNotFoundError:
    aho_corasick_c_lib = None
    logger.warning("Could not find the shared Aho-Corasick C library. Using Python implementation instead." +
                   " C lib is a few time faster so it's commanded to compile the code in c-lib.")

count_segments = count_segments_py if aho_corasick_c_lib is None else lambda nodes, segments: _count_segments_c(nodes, segments, aho_corasick_c_lib)
