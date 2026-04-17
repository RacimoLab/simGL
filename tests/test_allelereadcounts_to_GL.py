import pytest
import numpy as np
import simGL
from itertools import combinations_with_replacement


# --- Shape ---

def test_output_shape_diploid(arc):
    GL = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2)
    n_sites, n_ind, _ = arc.shape
    n_genotypes = len(list(combinations_with_replacement([0, 1, 2, 3], 2)))  # 10
    assert GL.shape == (n_sites, n_ind, n_genotypes)

def test_output_shape_haploid(arc):
    GL = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=1)
    assert GL.shape == (arc.shape[0], arc.shape[1], 4)


# --- Normalization ---

def test_normalized_has_one_zero_per_individual(arc):
    GL = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2, normalized=True)
    # Every site x individual must have exactly one GL == 0
    zeros_per_cell = (GL == 0.0).sum(axis=2)
    assert (zeros_per_cell == 1).all()

def test_unnormalized_all_nonnegative(arc):
    GL = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2, normalized=False)
    assert (GL >= 0).all()

def test_normalized_vs_unnormalized_consistent(arc):
    GL_raw  = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2, normalized=False)
    GL_norm = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2, normalized=True)
    np.testing.assert_allclose(GL_norm, simGL.normalize_GL(GL_raw))


# --- Correctness ---

def test_most_likely_genotype_matches_true(gm, ref_alt):
    # Simulate reads with zero error so every read perfectly reflects the true allele.
    # Compute GL with a very small but valid e (1e-6, ~Q60) — essentially noiseless.
    # After subsetting to the ref/alt allele pair, the called genotype (argmin) must
    # match the true genotype 100% of the time.
    ref, alt = ref_alt
    arc_noiseless = simGL.sim_allelereadcounts(
        gm, mean_depth=50., std_depth=5., e=0.0, ploidy=2,
        ref=ref, alt=alt, seed=0,
    )
    GL = simGL.allelereadcounts_to_GL(arc_noiseless, e=1e-6, ploidy=2, normalized=False)

    # Subset to biallelic (ref/alt) genotypes and normalize
    Ra = simGL.ref_alt_to_index(ref, alt)
    GL_sub = simGL.normalize_GL(simGL.subset_GL(GL, Ra, ploidy=2))

    # argmin gives 0 (hom ref), 1 (het), or 2 (hom alt) in the biallelic GL
    called = GL_sub.argmin(axis=2)

    # True genotype: sum of the two haplotypes per individual (0, 1, or 2)
    n_ind = gm.shape[1] // 2
    true_allele_counts = gm.reshape(gm.shape[0], n_ind, 2).sum(axis=2)
    assert (called == true_allele_counts).all()


# --- Input validation ---

def test_bad_arc_raises():
    with pytest.raises(TypeError):
        simGL.allelereadcounts_to_GL(np.zeros((10, 5, 3)), e=0.01, ploidy=2)

def test_bad_e_negative_raises(arc):
    with pytest.raises(TypeError):
        simGL.allelereadcounts_to_GL(arc, e=-0.1, ploidy=2)

def test_bad_e_zero_raises(arc):
    # e=0 is valid for simulation but undefined for GL computation
    with pytest.raises(TypeError):
        simGL.allelereadcounts_to_GL(arc, e=0.0, ploidy=2)

def test_bad_ploidy_raises(arc):
    with pytest.raises(TypeError):
        simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=0)
