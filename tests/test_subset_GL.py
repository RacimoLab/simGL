import pytest
import numpy as np
import simGL


# --- Shape ---

def test_output_shape_biallelic_diploid(GL, ref_alt):
    ref, alt = ref_alt
    Ra = simGL.ref_alt_to_index(ref, alt)
    GL_sub = simGL.subset_GL(GL, Ra, ploidy=2)
    # 2 alleles, diploid → 3 genotypes (hom-ref, het, hom-alt)
    assert GL_sub.shape == (GL.shape[0], GL.shape[1], 3)


# --- Correctness ---

def test_subset_values_match_full_GL():
    # Build a simple known case: 1 site, 1 individual, all-A reads, diploid
    # With ref=A(0) alt=C(1), the biallelic genotypes are AA(idx 0), AC(idx 1), CC(idx 4)
    # in the full 10-genotype GL. After subsetting we expect those three values in order.
    arc = np.array([[[30, 0, 0, 0]]])   # 30 A-reads, no errors
    GL_full = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2, normalized=False)
    alleles = np.array([[0, 1]])        # ref=A(0), alt=C(1)
    GL_sub  = simGL.subset_GL(GL_full, alleles, ploidy=2)

    # Genotype order in full GL for diploid: AA=0, AC=1, AG=2, AT=3, CC=4, ...
    expected = GL_full[:, :, [0, 1, 4]]
    np.testing.assert_array_equal(GL_sub, expected)

def test_unsorted_alleles_give_same_values_as_sorted(GL, ref_alt):
    # subset_GL should return genotypes in the order ref/alt regardless of allele index order.
    # Querying [ref, alt] vs [alt, ref] should give the same GL values (reordered correctly).
    ref, alt = ref_alt
    Ra       = simGL.ref_alt_to_index(ref, alt)     # e.g. [[2,3], [0,1], ...]
    Ra_flip  = Ra[:, ::-1].copy()                   # flip to [alt, ref]

    GL_sub      = simGL.subset_GL(GL, Ra,      ploidy=2)
    GL_sub_flip = simGL.subset_GL(GL, Ra_flip, ploidy=2)

    # Hom-ref (index 0) and hom-alt (index 2) should swap; het (index 1) stays the same
    np.testing.assert_array_equal(GL_sub[:, :, 0], GL_sub_flip[:, :, 2])
    np.testing.assert_array_equal(GL_sub[:, :, 1], GL_sub_flip[:, :, 1])
    np.testing.assert_array_equal(GL_sub[:, :, 2], GL_sub_flip[:, :, 0])

def test_subset_then_normalize_has_one_zero(GL, ref_alt):
    ref, alt = ref_alt
    Ra     = simGL.ref_alt_to_index(ref, alt)
    GL_sub = simGL.normalize_GL(simGL.subset_GL(GL, Ra, ploidy=2))
    zeros  = (GL_sub == 0.0).sum(axis=2)
    assert (zeros == 1).all()


# --- Input validation ---

def test_bad_alleles_per_site_shape_raises(GL):
    with pytest.raises(TypeError):
        simGL.subset_GL(GL, np.array([0, 1, 2]), ploidy=2)  # 1-D, should be 2-D

def test_alleles_per_site_wrong_n_sites_raises(GL, ref_alt):
    ref, alt = ref_alt
    Ra = simGL.ref_alt_to_index(ref, alt)
    with pytest.raises(TypeError):
        simGL.subset_GL(GL, Ra[:-1], ploidy=2)  # one fewer site than GL

def test_bad_ploidy_raises(GL, ref_alt):
    ref, alt = ref_alt
    Ra = simGL.ref_alt_to_index(ref, alt)
    with pytest.raises(TypeError):
        simGL.subset_GL(GL, Ra, ploidy=0)
