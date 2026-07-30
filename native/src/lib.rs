//! Native segment-counting kernel for PGSE.
//!
//! Counting semantics:
//!   * every occurrence of every segment is counted, including overlapping ones
//!     (`MatchKind::Standard` + overlapping iteration);
//!   * when a complement table is supplied (DNA), each segment is counted on both
//!     strands by also matching its reverse complement, non-canonical segments are
//!     skipped (their counts accrue to their canonical twin), and a palindrome is
//!     matched only once.

use aho_corasick::{AhoCorasick, MatchKind};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Reverse-complement a byte string using a 256-entry byte map, matching
/// `get_complement` in the C library (reverse the sequence, map each byte).
fn reverse_complement(seq: &[u8], table: &[u8; 256]) -> Vec<u8> {
    seq.iter().rev().map(|&b| table[b as usize]).collect()
}

/// A prebuilt automaton over a segment pool, reusable across many sequences.
#[pyclass]
struct SegmentMatcher {
    ac: AhoCorasick,
    /// PatternID -> index of the segment in the original pool. A segment and its
    /// reverse complement are two patterns that map to the same segment index.
    pattern_to_segment: Vec<u32>,
    n_segments: usize,
}

#[pymethods]
impl SegmentMatcher {
    /// Build the automaton from the segment pool.
    ///
    /// * `segments` – the pool, already case-normalised by the caller.
    /// * `complement` – optional 256-byte reverse-complement table. `None` disables
    ///   canonicalisation (the usual case for non-DNA alphabets).
    #[new]
    #[pyo3(signature = (segments, complement=None))]
    fn new(segments: Vec<String>, complement: Option<Vec<u8>>) -> PyResult<Self> {
        let n_segments = segments.len();

        let table: Option<[u8; 256]> = match complement {
            None => None,
            Some(v) => {
                if v.len() != 256 {
                    return Err(PyValueError::new_err(
                        "complement table must have exactly 256 entries",
                    ));
                }
                let mut t = [0u8; 256];
                t.copy_from_slice(&v);
                Some(t)
            }
        };

        // Roughly one pattern per segment (two when a complement is added), so
        // reserve accordingly to avoid reallocations on large pools.
        let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(n_segments);
        let mut pattern_to_segment: Vec<u32> = Vec::with_capacity(n_segments);

        for (i, seg) in segments.iter().enumerate() {
            let s = seg.as_bytes();
            // An empty segment can never match; the C library drops it too.
            if s.is_empty() {
                continue;
            }
            match &table {
                None => {
                    patterns.push(s.to_vec());
                    pattern_to_segment.push(i as u32);
                }
                Some(t) => {
                    let comp = reverse_complement(s, t);
                    // is_canonical: keep only segments that are <= their complement.
                    // Non-canonical segments are counted under their canonical twin.
                    if s > comp.as_slice() {
                        continue;
                    }
                    patterns.push(s.to_vec());
                    pattern_to_segment.push(i as u32);
                    // Non-palindromes are also matched via their reverse complement,
                    // which is how both strands are counted from the forward text.
                    if comp.as_slice() != s {
                        patterns.push(comp);
                        pattern_to_segment.push(i as u32);
                    }
                }
            }
        }

        let ac = AhoCorasick::builder()
            .match_kind(MatchKind::Standard)
            .build(&patterns)
            .map_err(|e| PyValueError::new_err(format!("failed to build automaton: {e}")))?;

        Ok(SegmentMatcher {
            ac,
            pattern_to_segment,
            n_segments,
        })
    }

    #[getter]
    fn n_segments(&self) -> usize {
        self.n_segments
    }

    /// Count segment occurrences across a batch of samples.
    ///
    /// * `samples` – one entry per sample, each a list of node strings (contigs),
    ///   already case-normalised and restricted to the alphabet by the caller.
    /// * `threads` – size of the rayon pool to use (0 = rayon default). Bounding this
    ///   lets a single Ray task use exactly its allotted cores.
    ///
    /// Returns the CSR triplet `(indptr, indices, data)` for the batch: `indptr`
    /// (int64, len `samples+1`), `indices` (int32, segment/column ids) and `data`
    /// (uint32, counts). Column ids are the original pool positions and are sorted
    /// within each row.
    #[pyo3(signature = (samples, threads=0))]
    fn count_csr<'py>(
        &self,
        py: Python<'py>,
        samples: Vec<Vec<String>>,
        threads: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<u32>>,
    )> {
        let ac = &self.ac;
        let p2s = &self.pattern_to_segment;

        // The parallel section touches only Rust-owned data (the automaton and the
        // moved-in samples), so it is safe to drop the GIL and let rayon run flat out.
        let rows: Vec<(Vec<i32>, Vec<u32>)> = py.allow_threads(|| {
            let count = || {
                samples
                    .par_iter()
                    .map(|nodes| count_one(ac, p2s, nodes))
                    .collect()
            };
            if threads > 0 {
                // A scoped pool keeps us to the cores this task was given.
                match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                    Ok(pool) => pool.install(count),
                    Err(_) => count(),
                }
            } else {
                count()
            }
        });

        // Concatenate per-row (index, data) into CSR arrays.
        let n = rows.len();
        let total: usize = rows.iter().map(|(idx, _)| idx.len()).sum();
        let mut indptr: Vec<i64> = Vec::with_capacity(n + 1);
        indptr.push(0);
        let mut indices: Vec<i32> = Vec::with_capacity(total);
        let mut data: Vec<u32> = Vec::with_capacity(total);
        for (idx, val) in &rows {
            indices.extend_from_slice(idx);
            data.extend_from_slice(val);
            indptr.push(indices.len() as i64);
        }

        Ok((
            indptr.into_pyarray(py),
            indices.into_pyarray(py),
            data.into_pyarray(py),
        ))
    }
}

/// Count all segment occurrences in one sample's nodes and return the sorted
/// (segment id, count) pairs as parallel vectors.
fn count_one(ac: &AhoCorasick, p2s: &[u32], nodes: &[String]) -> (Vec<i32>, Vec<u32>) {
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for node in nodes {
        for m in ac.find_overlapping_iter(node) {
            let seg = p2s[m.pattern().as_usize()];
            *counts.entry(seg).or_insert(0) += 1;
        }
    }

    let mut pairs: Vec<(u32, u32)> = counts.into_iter().collect();
    // CSR wants column indices sorted within a row.
    pairs.sort_unstable_by_key(|&(seg, _)| seg);

    let mut indices = Vec::with_capacity(pairs.len());
    let mut data = Vec::with_capacity(pairs.len());
    for (seg, c) in pairs {
        indices.push(seg as i32);
        data.push(c);
    }
    (indices, data)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SegmentMatcher>()?;
    Ok(())
}
