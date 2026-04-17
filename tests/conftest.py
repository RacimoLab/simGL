import pytest
import numpy as np
import msprime
import simGL


def filter_biallelic(tree_sequence):
    """
    Returns a boolean mask selecting only biallelic sites from a tskit tree sequence.

    simGL requires genotype matrices with values 0 and 1 only. Tree sequences produced
    by msprime can contain multiallelic sites (genotype values > 1) when multiple
    mutations hit the same position. This helper identifies and removes those sites.

    Parameters
    ----------
    tree_sequence : tskit.TreeSequence

    Returns
    -------
    mask : numpy.ndarray, dtype bool
        Boolean array of shape (n_sites,). True where the site is biallelic.
    """
    gm_full = tree_sequence.genotype_matrix()
    return gm_full.max(axis=1) == 1


@pytest.fixture(scope="session")
def tree_sequence():
    ts = msprime.sim_ancestry(
        samples=5, ploidy=2, sequence_length=10_000,
        recombination_rate=1e-8, population_size=10_000, random_seed=42,
    )
    return msprime.sim_mutations(ts, rate=1e-3, random_seed=42)


@pytest.fixture(scope="session")
def gm(tree_sequence):
    mask = filter_biallelic(tree_sequence)
    return tree_sequence.genotype_matrix()[mask].astype(int)


@pytest.fixture(scope="session")
def ref_alt(tree_sequence):
    mask = filter_biallelic(tree_sequence)
    ref = np.array([v.site.ancestral_state for v in tree_sequence.variants()])[mask]
    alt = np.array([v.site.mutations[0].derived_state for v in tree_sequence.variants()])[mask]
    return ref, alt


@pytest.fixture(scope="session")
def pos(tree_sequence):
    mask = filter_biallelic(tree_sequence)
    return np.array([int(v.site.position) for v in tree_sequence.variants()])[mask]


@pytest.fixture(scope="session")
def arc(gm, ref_alt):
    ref, alt = ref_alt
    return simGL.sim_allelereadcounts(
        gm, mean_depth=10., std_depth=2., e=0.01, ploidy=2,
        ref=ref, alt=alt, seed=42,
    )


@pytest.fixture(scope="session")
def GL(arc):
    return simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2, normalized=False)
