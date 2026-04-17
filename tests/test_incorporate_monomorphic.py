import pytest
import numpy as np
import simGL


# --- Shape ---

def test_output_shape(gm, pos):
    start = int(pos[0])
    end   = int(pos[-1]) + 1
    gm2 = simGL.incorporate_monomorphic(gm, pos, start, end)
    assert gm2.shape == (end - start, gm.shape[1])

def test_monomorphic_sites_are_zero(gm, pos):
    start = 0
    end   = int(pos[-1]) + 1
    gm2 = simGL.incorporate_monomorphic(gm, pos, start, end)
    monomorphic_mask = np.ones(gm2.shape[0], dtype=bool)
    monomorphic_mask[pos - start] = False
    assert (gm2[monomorphic_mask] == 0).all()

def test_polymorphic_sites_preserved(gm, pos):
    start = int(pos[0])
    end   = int(pos[-1]) + 1
    gm2 = simGL.incorporate_monomorphic(gm, pos, start, end)
    np.testing.assert_array_equal(gm2[pos - start], gm)


# --- Input validation ---

def test_start_too_large_raises(gm, pos):
    with pytest.raises(TypeError):
        simGL.incorporate_monomorphic(gm, pos, start=int(pos[0]) + 1, end=int(pos[-1]) + 1)

def test_end_too_small_raises(gm, pos):
    with pytest.raises(TypeError):
        simGL.incorporate_monomorphic(gm, pos, start=int(pos[0]), end=int(pos[-1]) - 1)
