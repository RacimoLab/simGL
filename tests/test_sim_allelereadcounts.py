import pytest
import numpy as np
import simGL


# --- Shape and dtype ---

def test_output_shape(arc, gm):
    # (sites, individuals, 4 alleles) — individuals = haplotypes / ploidy
    assert arc.shape == (gm.shape[0], gm.shape[1] // 2, 4)

def test_output_nonnegative(arc):
    assert (arc >= 0).all()

def test_output_integer(arc):
    assert np.issubdtype(arc.dtype, np.integer)


# --- Reproducibility ---

def test_seed_reproducibility(gm, ref_alt):
    ref, alt = ref_alt
    arc1 = simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=2., e=0.01, ploidy=2, ref=ref, alt=alt, seed=99)
    arc2 = simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=2., e=0.01, ploidy=2, ref=ref, alt=alt, seed=99)
    np.testing.assert_array_equal(arc1, arc2)

def test_different_seeds_differ(gm, ref_alt):
    ref, alt = ref_alt
    arc1 = simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=2., e=0.01, ploidy=2, ref=ref, alt=alt, seed=1)
    arc2 = simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=2., e=0.01, ploidy=2, ref=ref, alt=alt, seed=2)
    assert not np.array_equal(arc1, arc2)


# --- Coverage ---

def test_mean_coverage_approx(gm, ref_alt):
    # Mean total reads per individual per site should be ~2 * mean_depth (ploidy=2)
    ref, alt = ref_alt
    arc = simGL.sim_allelereadcounts(gm, mean_depth=20., std_depth=1., e=0.001, ploidy=2, ref=ref, alt=alt, seed=0)
    mean_cov = arc.sum(axis=2).mean()
    assert 35 < mean_cov < 45   # expect ~40 (2 haplotypes x 20x)

def test_array_depth_input(gm, ref_alt):
    # Passing a depth array directly should bypass gamma sampling
    ref, alt = ref_alt
    depth_array = np.full(gm.shape[1], 10.0)
    arc = simGL.sim_allelereadcounts(gm, mean_depth=depth_array, e=0.01, ploidy=2, ref=ref, alt=alt, seed=0)
    assert arc.shape == (gm.shape[0], gm.shape[1] // 2, 4)


# --- Error rate ---

def test_error_rate(gm):
    # With no ref/alt provided, ref="A" alt="C" for all sites.
    # Use a homozygous-reference matrix (all 0s) so the true allele is always "A".
    # With high coverage and error rate e, fraction of non-A reads should be ~e.
    n_sites, n_hap = 50, 10
    gm_ref = np.zeros((n_sites, n_hap), dtype=int)
    arc = simGL.sim_allelereadcounts(gm_ref, mean_depth=500., std_depth=10., e=0.1, ploidy=1, seed=0)
    total = arc.sum()
    errors = arc[:, :, 1:].sum()   # all non-"A" reads are errors
    assert abs(errors / total - 0.1) < 0.02


# --- Input validation ---

def test_bad_gm_raises():
    with pytest.raises(TypeError):
        simGL.sim_allelereadcounts(np.array([1, 2, 3]), mean_depth=10., std_depth=1., e=0.01, ploidy=2)

def test_bad_ploidy_raises(gm):
    with pytest.raises(TypeError):
        simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=1., e=0.01, ploidy=3)  # not divisible

def test_bad_e_raises(gm):
    with pytest.raises(TypeError):
        simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=1., e=1.5, ploidy=2)

def test_bad_ref_shape_raises(gm, ref_alt):
    ref, alt = ref_alt
    with pytest.raises(TypeError):
        simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=1., e=0.01, ploidy=2, ref=ref[:-1], alt=alt[:-1])
