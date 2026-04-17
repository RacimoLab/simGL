import pytest
import numpy as np
import simGL


# --- Encoding ---

def test_all_bases_encoded_correctly():
    ref = np.array(["A", "C", "G", "T"])
    alt = np.array(["T", "G", "C", "A"])
    result = simGL.ref_alt_to_index(ref, alt)
    np.testing.assert_array_equal(result, [[0, 3], [1, 2], [2, 1], [3, 0]])

def test_output_shape(ref_alt):
    ref, alt = ref_alt
    result = simGL.ref_alt_to_index(ref, alt)
    assert result.shape == (len(ref), 2)

def test_first_col_is_ref_second_is_alt(ref_alt):
    ref, alt = ref_alt
    result = simGL.ref_alt_to_index(ref, alt)
    base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    expected_ref = np.array([base_to_int[b] for b in ref])
    expected_alt = np.array([base_to_int[b] for b in alt])
    np.testing.assert_array_equal(result[:, 0], expected_ref)
    np.testing.assert_array_equal(result[:, 1], expected_alt)


# --- Input validation ---

def test_non_array_raises():
    with pytest.raises(TypeError):
        simGL.ref_alt_to_index(["A", "C"], np.array(["T", "G"]))

def test_mismatched_lengths_raises():
    with pytest.raises(TypeError):
        simGL.ref_alt_to_index(np.array(["A", "C"]), np.array(["T"]))

def test_invalid_base_raises():
    with pytest.raises(TypeError):
        simGL.ref_alt_to_index(np.array(["A", "X"]), np.array(["T", "G"]))
