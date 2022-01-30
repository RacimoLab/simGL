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


######################################MOI#############################

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