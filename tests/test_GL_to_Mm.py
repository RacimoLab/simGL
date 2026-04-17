import pytest
import numpy as np
import simGL


# --- Shape and values ---

def test_output_shape(GL):
    Mm = simGL.GL_to_Mm(GL, ploidy=2)
    assert Mm.shape == (GL.shape[0], 2)

def test_output_values_in_range(GL):
    Mm = simGL.GL_to_Mm(GL, ploidy=2)
    assert ((Mm >= 0) & (Mm <= 3)).all()

def test_alleles_are_sorted(GL):
    # GL_to_Mm returns allele indices sorted in ascending order
    Mm = simGL.GL_to_Mm(GL, ploidy=2)
    assert (Mm[:, 0] <= Mm[:, 1]).all()

def test_major_minor_are_different(GL):
    # The two alleles must be distinct (major ≠ minor)
    Mm = simGL.GL_to_Mm(GL, ploidy=2)
    assert (Mm[:, 0] != Mm[:, 1]).all()


# --- Correctness ---

def test_identified_alleles_match_true(gm, ref_alt):
    # GL_to_Mm requires polymorphic sites — filter out fixed sites (alt_freq 0 or 1)
    # before calling, as the second allele is unidentifiable from GL at fixed sites.
    ref, alt = ref_alt
    polymorphic = (gm.mean(axis=1) > 0) & (gm.mean(axis=1) < 1)
    gm_poly, ref_poly, alt_poly = gm[polymorphic], ref[polymorphic], alt[polymorphic]

    arc_noiseless = simGL.sim_allelereadcounts(
        gm_poly, mean_depth=50., std_depth=5., e=0.0, ploidy=2,
        ref=ref_poly, alt=alt_poly, seed=0,
    )
    GL = simGL.allelereadcounts_to_GL(arc_noiseless, e=1e-6, ploidy=2, normalized=False)
    Mm = simGL.GL_to_Mm(GL, ploidy=2)

    # ref_alt_to_index gives (sites, 2) — sort ascending to match GL_to_Mm output order
    Ra = simGL.ref_alt_to_index(ref_poly, alt_poly)
    Ra_sorted = np.sort(Ra, axis=1)

    np.testing.assert_array_equal(Mm, Ra_sorted)


# --- Input validation ---

def test_bad_ploidy_raises(GL):
    with pytest.raises(TypeError):
        simGL.GL_to_Mm(GL, ploidy=0)

def test_bad_GL_shape_raises():
    with pytest.raises(TypeError):
        simGL.GL_to_Mm(np.zeros((10, 5)), ploidy=2)

def test_GL_ploidy_mismatch_raises(GL):
    # GL has 10 genotypes (diploid), but ploidy=1 expects 4
    with pytest.raises(TypeError):
        simGL.GL_to_Mm(GL, ploidy=1)
