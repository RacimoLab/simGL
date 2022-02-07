import msprime
import simGL
import tskit
import numpy as np


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
    ts  = sim_ts(n=num_individuals)
    arc = simGL.sim_allelereadcounts(gm = ts.genotype_matrix(), mean_depth = 15, std_depth = 2.5, e = 0.05, ploidy = 2, seed = 1234)
    assert arc.shape == (ts.num_sites, num_individuals, 4)

def test_allelereadcounts_to_GL_inputs():
    ts = sim_ts()
    ref = np.array([v.site.ancestral_state for v in ts.variants()])
    alt = np.array([v.site.mutations[0].derived_state for v in ts.variants()])
    # gm is :
    #     - not a numpy array
    #     - is a numpy array with only one dimention
    #     - is a numpy array with values different than 1 or 0
    for gm in [2, np.array([0, 1, 1, 0]), np.array([[0, 3], [1, 0]])]:
        try:
            simGL.sim_allelereadcounts(gm = gm, mean_depth = 30, e = 0.05, ploidy = 2, seed = 1234, std_depth = 5, ref = ref, alt = alt)
        except TypeError:
            pass
    gm = ts.genotype_matrix()
    # mean_depth is:
    #     - string
    #     - negative value
    #     - numpy array with more than 1 dimention
    #     - numpy array with 1 dimention and negative or 0 values
    #     - numpy array with 1 dimention and positive values with a different size than gm.shape[1]
    for mean_depth in ["30", -1, np.array([[15], [15]]), np.array([15, 0]), np.full(gm.shape[1]-1, 15)]:
        try:
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = 0.05, ploidy = 2, seed = 1234, std_depth = 5, ref = ref, alt = alt)
        except TypeError:
            pass
    # std_depth is:
    # - string
    # - negative number
    # - numpy array
    for std_depth in ["5", -4, np.array([3])]:
        try:
            print(isinstance(std_depth, (int, float)))
            simGL.sim_allelereadcounts(gm = gm, mean_depth = 30, e = 0.05, ploidy = 2, seed = 1234, std_depth = std_depth, ref = ref, alt = alt)
        except TypeError:
            pass

def test_allelereadcounts_to_GL():
    # TODO: proper tests
    num_individuals = 5
    ts  = sim_ts(n=num_individuals)
    arc = simGL.sim_allelereadcounts(gm = ts.genotype_matrix(), mean_depth = 15, std_depth = 2.5, e = 0.05, ploidy = 2, seed = 1234)
    GL  = simGL.allelereadcounts_to_GL(arc = arc, e = 0.05, ploidy = 2)
    assert GL.shape == (ts.num_sites, num_individuals, 10)

def allelereadcounts_to_GL_forloop(Rg, e = 0.05):
    GL = []
    
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                GL.append(-np.log(np.power(((1-e)/2 + (1-e)/2), Rg[:, :, i]) * 
                                  np.power(((e/3)/2 + (e/3)/2), Rg.sum(axis = 2)-Rg[:, :, i])))
            else:
            
                GL.append(-np.log(np.power(((1-e)/2 + (e/3)/2), Rg[:, :, i]+Rg[:, :, j]) * 
                                  np.power(((e/3)/2 + (e/3)/2), Rg.sum(axis = 2)-Rg[:, :, i]-Rg[:, :, j])))

    GL = np.array(GL).transpose(1, 2, 0)
    return GL - GL.min(axis = 2).reshape(GL.shape[0], GL.shape[1], 1)

def create_fasta(length = 100_000, filename = "tmp/fasta.fa"):
    with open(filename, "w") as fasta:
        fasta.write(">1\n")
        for i in range(0, length, 50):
            fasta.write("{}\n".format("A"*50))
    
def read_angsd_gl(file = "tmp/angsdput.glf"):
    angsd_gl = []
    with open(file, "r") as f:
        for line in f:
            angsd_gl.append(np.array(line.strip().split()[2:]).reshape(5, 10).astype(np.float64).tolist())
    return -np.array(angsd_gl)

def msqrd(a, b):
    return np.power(a-b, 2).sum()/a.size