from pgse.algos.aho_corasick_c import AhoCorasickC as _AhoCorasickC
from pgse.algos.aho_corasick_py import AhoCorasickPy as _AhoCorasickPy
from pgse.log import logger

try:
    aho_corasick = _AhoCorasickC()
except FileNotFoundError as _e:
    aho_corasick = _AhoCorasickPy()
    logger.warning(f"Could not use the shared Aho-Corasick C library ({_e}). Using Python implementation instead." +
                   " C lib is a few time faster so it's recommended to compile the code in c_lib.")
