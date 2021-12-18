import msprime

import simGL


def sim_ts(seeds=(1, 2), n=5):
    """Generate a tree sequence for testing."""
    ts = msprime.sim_ancestry(
        population_size=10_000,
        samples=n,
        sequence_length=100_000,
        recombination_rate=1.25e-8,
        random_seed=seeds[0],
        record_provenance=False,
    )
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=seeds[1])
    return ts


def test_sim_allelereadcounts():
    # TODO: proper tests
    num_individuals = 5
    ts = sim_ts(n=num_individuals)
    ac = simGL.sim_allelereadcounts(ts, seed=1234)
    assert ac.shape == (ts.num_sites, num_individuals, 4)


def test_allelereadcounts_to_GL():
    # TODO: proper tests
    num_individuals = 5
    ts = sim_ts(n=num_individuals)
    ac = simGL.sim_allelereadcounts(ts, seed=1234)
    GL = simGL.allelereadcounts_to_GL(ac)
    assert GL.shape == (10, ts.num_sites, num_individuals)
