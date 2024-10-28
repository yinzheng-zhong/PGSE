import ctypes

import numpy as np


def load_lib():
    lib = None

    possible_locations = [
        '../../c-lib/aho_corasick.so',
        '../c-lib/src/algos/aho_corasick.so',
        'c-lib/src/algos/aho_corasick.so',
    ]
    # try to load the library
    for location in possible_locations:
        try:
            lib = ctypes.CDLL(location)
            break
        except OSError:
            pass

    if lib is None:
        raise FileNotFoundError("Could not find the shared library")

    lib.count_segments.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),  # nodes
        ctypes.c_int,  # num_nodes
        ctypes.POINTER(ctypes.c_char_p),  # segments
        ctypes.c_int,  # num_segments
        ctypes.POINTER(ctypes.c_int)  # result_counts
    ]
    lib.count_segments.restype = None

    return lib

def count_segments(nodes, segments, lib):
    num_nodes = len(nodes)
    num_segments = len(segments)

    # Create arrays of c_char_p
    node_array = (ctypes.c_char_p * num_nodes)(*(node.encode('utf-8') for node in nodes))
    segment_array = (ctypes.c_char_p * num_segments)(*(seg.encode('utf-8') for seg in segments))

    # Prepare result array
    result_counts = (ctypes.c_int * num_segments)()

    # Call the C function
    lib.count_segments(
        node_array, ctypes.c_int(num_nodes),
        segment_array, ctypes.c_int(num_segments),
        result_counts
    )

    # Convert result to NumPy array
    seq_count = np.ctypeslib.as_array(result_counts, shape=(num_segments,))

    return seq_count