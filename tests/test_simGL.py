import msprime
import simGL
import tskit
import numpy as np
import pytest


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
    #Check that the output is correct according to gm dimentions and according to ploidy
    for ploidy in [2, 4]:
        num_individuals = 5
        ts  = sim_ts(n=int(5*ploidy/2))
        arc = simGL.sim_allelereadcounts(gm = ts.genotype_matrix(), mean_depth = 15, std_depth = 2.5, e = 0.05, ploidy = ploidy, seed = 1234)
        assert arc.shape == (ts.num_sites, num_individuals, 4)

def test_input_checks():
    #Checks that the functions that raise errors correctly do so when the input is wrong
    ts = sim_ts()
    # gm is :
    #     - not a numpy array
    #     - is a numpy array with only one dimention
    #     - is a numpy array with values different than 1 or 0
    for gm in [2, np.array([0, 1, 1, 0]), np.array([[0, 3], [1, 0]])]:
        with pytest.raises(TypeError, match=r"^Incorrect gm format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = 30, e = 0.05, ploidy = 2, seed = 1234, std_depth = 5, ref = None, alt = None, depth_type = "independent", read_length = None, start = None, end = None, pos = None) 
    gm = ts.genotype_matrix()
    # mean_depth is:
    #     - string
    #     - negative value
    #     - numpy array with more than 1 dimention
    #     - numpy array with 1 dimention and negative or 0 values
    #     - numpy array with 1 dimention and positive values with a different size than gm.shape[1]
    for mean_depth in ["30", -1, np.array([[15], [15]]), np.array([15, 0]), np.full(gm.shape[1]-1, 15)]:
        with pytest.raises(TypeError, match=r"^Incorrect mean_depth format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = 0.05, ploidy = 2, seed = 1234, std_depth = 5, ref = None, alt = None, depth_type = "independent", read_length = None, start = None, end = None, pos = None)
    mean_depth = 15
    # std_depth is:
    # - string
    # - negative number
    # - numpy array
    for std_depth in ["5", -4, np.array([3])]:
        with pytest.raises(TypeError, match=r"^Incorrect std_depth format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = 0.05, ploidy = 2, seed = 1234, std_depth = std_depth, ref = None, alt = None, depth_type = "independent", read_length = None, start = None, end = None, pos = None)
    std_depth = 5
    # e is:
    # - string
    # - > 1
    # - < 1
    # - numpy array
    for e in ["5", -4, 3, np.array([0.5])]:
        with pytest.raises(TypeError, match=r"^Incorrect e format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = e, ploidy = 2, seed = 1234, std_depth = std_depth, ref = None, alt = None, depth_type = "independent", read_length = None, start = None, end = None, pos = None)
    e = 0.05
    # ploidy is:
    # - string
    # - float < 1
    # - gm.shape[1] not multiple of ploidy
    # - numpy array
    for ploidy in ["5", 0.5, np.array([1])]:
        with pytest.raises(TypeError, match=r"^Incorrect ploidy format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = e, ploidy = ploidy, seed = 1234, std_depth = std_depth, ref = None, alt = None, depth_type = "independent", read_length = None, start = None, end = None, pos = None)
    # - gm.shape[1] not multiple of ploidy
    with pytest.raises(TypeError, match=r"^Incorrect ploidy and/or gm format: .*"):
        simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = e, ploidy = gm.shape[1]+1, seed = 1234, std_depth = std_depth, ref = None, alt = None, depth_type = "independent", read_length = None, start = None, end = None, pos = None)
    ploidy = 2
    # ref and alt are:
    # - not the same length
    # - one None and the other a numpy array
    # - have not a single dimention
    # - it does not contain an allowed basepair
    # - they don't have the same size as gm.shape[1]
    # - one of them is a python list
    ref = np.array([v.site.ancestral_state for v in ts.variants()])
    alt = np.array([v.site.mutations[0].derived_state for v in ts.variants()])
    ref_tt = np.copy(ref)
    ref_tt[0] = "X"
    for ref_t, alt_t in zip([ref,      None, ref.reshape(ref.shape[0], 1), ref_tt, np.array(["A", "C"]), ref],
                            [alt[:-2], alt,  alt,                          alt,    np.array(["C", "A"]), alt.tolist()]):
        with pytest.raises(TypeError, match=r"^Incorrect ref and/or alt format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = e, ploidy = ploidy, seed = 1234, std_depth = std_depth, ref = ref_t, alt = alt_t, depth_type = "independent", read_length = None, start = None, end = None, pos = None)
    # depth_type is not one of the two strings: independent or linked:
    for depth_type in [0, None, "asdfa"]:
        with pytest.raises(TypeError, match=r"^Incorrect depth_type format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = e, ploidy = ploidy, seed = 1234, std_depth = std_depth, ref = ref, alt = alt, depth_type = depth_type, read_length = None, start = None, end = None, pos = None)
    depth_type = "linked"
    # read_length is:
    #     - not a positive integer above 0
    for read_length in [0, "asdfa", 5.4]:
        with pytest.raises(TypeError, match=r"^Incorrect read_length format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = e, ploidy = ploidy, seed = 1234, std_depth = std_depth, ref = ref, alt = alt, depth_type = depth_type, read_length = read_length, start = None, end = None, pos = None)
    read_length = 100
    # start is:
    #     - not a positive integer or 0 smaller than the minimum coordinate position 
    pos = np.array([int(v.site.position) for v in ts.variants()])
    end = 1_000_000
    for start in ["asdfa", 5.4, pos[0]+1]:
        with pytest.raises(TypeError, match=r"^Incorrect start format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = e, ploidy = ploidy, seed = 1234, std_depth = std_depth, ref = ref, alt = alt, depth_type = depth_type, read_length = read_length, start = start, end = end, pos = pos)
    start = 0
    #pos[0]+1
    # end is:
    #     - not a positive integer greater than the maximum coordinate position
    for end in [0, "asdfa", 5.4, pos[-1]-11]:
        with pytest.raises(TypeError, match=r"^Incorrect end format: .*"):
            simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = e, ploidy = ploidy, seed = 1234, std_depth = std_depth, ref = ref, alt = alt, depth_type = depth_type, read_length = read_length, start = start, end = end, pos = pos)
    end = 1_000_000
    # pos is :
    #     - not a numpy array
    #     - has values < 0
    #     - a numpy array with more than one dimention
    #     - of a different size than first dimention of gm
    #     - containing int values
    #     - unsorted       
    for t_pos in [2,  np.array([-4, -3, 0, 1]), np.arange(10).reshape(10//2, 2), 
                np.arange(gm.shape[0]-1), np.random.random_sample((gm.shape[0], )), 
                np.array(np.arange(gm.shape[0])[1:].tolist() + [0])]:
        with pytest.raises(TypeError, match=r"^Incorrect pos format: .*"):
            simGL.incorporate_monomorphic(gm = gm, pos = t_pos, start = 0, end = 10_000_000)
    pos = np.array([int(v.site.position) for v in ts.variants()])
    # start is :
    #     - < 0
    #     - bigger than start position
    for start in ["3", np.array([2]), -3, pos[-1]]:
        print(start)
        with pytest.raises(TypeError, match=r"^Incorrect start format: .*"):
            simGL.incorporate_monomorphic(gm = gm, pos = pos, start = start, end = 10_000_000)
    start = 0
    # end is :
    #     - <= 0
    #     - > max(pos) or > pos[-1]
    for end in ["3", np.array([2]), pos[0]]:
        with pytest.raises(TypeError, match=r"^Incorrect end format: .*"):
            simGL.incorporate_monomorphic(gm = gm, pos = pos, start = start, end = end)
    end = pos[-1]+1000
    # arc is :
    #     - not a numpy array (is a list)
    #     - of two dimentions
    #     - of three dimentions but last has size != 4
    for arc in [[3], np.arange(2*4).reshape(2, 4), np.arange(2*2*5).reshape(2, 2, 5)]:
        with pytest.raises(TypeError, match=r"^Incorrect arc format: .*"):
            simGL.allelereadcounts_to_GL(arc, e, ploidy)
    arc = simGL.sim_allelereadcounts(gm = gm, mean_depth = mean_depth, e = e, ploidy = ploidy, seed = 1234, std_depth = std_depth, ref = ref, alt = alt)
    # GL is :
    #     - not a numpy array (is a list)
    #     - of two dimentions
    for GL in [[3], np.arange(2*4).reshape(2, 4)]:
        with pytest.raises(TypeError, match=r"^Incorrect GL format: .*"):
            simGL.GL_to_Mm(GL, ploidy)
    #     - dimentions does not corresponds with ploidy
    GL = simGL.allelereadcounts_to_GL(arc, e, ploidy)
    with pytest.raises(TypeError, match=r"^Incorrect ploidy format or GL format: .*"):
        simGL.GL_to_Mm(GL, ploidy+1)
    # alleles_per_site is :
    #     - not a numpy array (is a list)
    #     - of one dimention
    #     - of three dimentions
    for alleles_per_site in [[3], np.array([3]), np.arange(2*4*3).reshape(2, 4, 3)]:
        with pytest.raises(TypeError, match=r"^Incorrect alleles_per_site format: .*"):
            simGL.subset_GL(GL, alleles_per_site, ploidy)

def test_allelereadcounts_to_GL():
    # TODO: proper tests
    num_individuals = 5
    ts  = sim_ts(n=num_individuals)
    arc = simGL.sim_allelereadcounts(gm = ts.genotype_matrix(), mean_depth = 15, std_depth = 2.5, e = 0.05, ploidy = 2, seed = 1234)
    GL  = simGL.allelereadcounts_to_GL(arc = arc, e = 0.05, ploidy = 2)
    assert GL.shape == (ts.num_sites, num_individuals, 10)

def test_Mm_same_as_refalt():
    # TODO: Make a test that checks that for a big number of coverage the obtained alleles for maximum and minimum is the same as reference and alternative alleles.
    pass

def test_GL_has_zero():
    # TODO: Check that all sites for all individuals have a GL that is equal to 0.
    pass

def test_subset_GL_has_zero():
    # TODO: Check that all sites for all individuals have a GL that is equal to 0.
    pass

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