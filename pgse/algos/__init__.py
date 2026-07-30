from pgse.algos.aho_corasick_py import AhoCorasickPy as _AhoCorasickPy

# Segment counting for the dataset is done by the native Rust kernel (pgse._native,
# see pgse.algos.native_counter), which builds the automaton once and shares it
# across threads. This pure-Python Aho-Corasick remains as the reference
# implementation and the fallback used when the native extension is unavailable
# (e.g. a source install without a Rust toolchain).
aho_corasick = _AhoCorasickPy()
