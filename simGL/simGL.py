import numpy as np
import tskit
from itertools import combinations_with_replacement

def samples_order(ts):
    so = []
    for ind in ts.individuals():
        [so.append(nod) for nod in ind.nodes]
    return np.array(so)

def extract_genotype_matrix(data):
    if type(data) == np.ndarray and len(data.shape) == 2:
        return data
    elif type(data) == tskit.trees.TreeSequence:
        return data.genotype_matrix()[:, samples_order(data)]
    else:
        sys.exit("Incorrect data format")

def depth_per_haplotype(rng, mean_depth, std_depth, n_hap, ploidy):
    if (type(mean_depth) == int or type(mean_depth) == float) and mean_depth > 0.0:
        if type(std_depth) == int or type(std_depth) == float:
            DPh = []
            while len(DPh) < n_hap:
                dp = rng.normal(loc = mean_depth/ploidy, scale = std_depth, size=1)[0]
                if dp > 0:
                    DPh.append(dp)
            return DPh
    elif type(mean_depth) == np.ndarray and len(mean_depth.shape) == 1 and mean_depth.shape[0] == n_hap and (mean_depth > 0).sum() == n_hap:
        return mean_depth
    else:
        raise Exception('Incorrect mean_depth format')
        
        
def sim_allelereadcounts(data, mean_depth = 30., std_depth = 5., e = 0.05, ploidy = 2, seed = 1234):
    '''
    Def:
        Function to simulate read counts for alleles given a tree sequence data from diploid simulated individuals (2samples = ind) 
        or genotype matrix and extra information for the haplotype samples.
    Input:
        - gm         : Genotype matrix in numpy format with shape (SNPs, samples). It is assumed that the array is sorted
                       according to a individual order such that consecutive columns (e.g., gm[:, 0] and gm[:, 1]) 
                       correspond to the same individual.
        - mean_depth : Two inputs are possible:
                            + float > 0 with the mean depth per sample. The mean depth for every sample haplotype 
                              is going to be sampled from a normal distribution with mean = mean_depth and std = std_depth.
                              Consider that since this script assumes that individuals are diploid, if the user intends to 
                              simulate a coverage of 30X per individual, the argument should be set to 15.
                            + numpy array with shape (samples, ) with the mean depth per haplotype sample. All values must 
                              be > 0. The order of the values is going to be associated to the list of individual's samples
                              provided by ts.individuals() (in case a tree sequence data is provided) or the order of the
                              samples in the genotype matrix (if a genotype matrix is provided in the input data).
                              If the intended coverage per site for a given individual is 30, since the coverage is 
                              given per haplotype, it should be indicated consecutively and half the individual coverage (15 
                              and 15 or 14 and 16).
        - std_depth  : float that corresponds to the standard deviation of the normal distribution from which coverages are
                       going to be sampled. This value will only be used if a float value is inputet for mean_depth. 
        - e          : float between 0 and 1 representing the error rate per base per read per site. This probability is
                       assumed to be constant.
        - ploidy     : int with the number of haplotypic sequences per individual
        - seed       : integer from which the numpy rng will be drawn. 
    Output:
        - Rg         : numpy array with dimentions (SNP, individual, alleles) so that each value corresponds to the number of
                       reads with a particular allele for a SNP position and a diploid individual. The index of the 3rd dimention
                       corresponds to 1 : A and ancestral allele, 2 : C and derived allele, 3 : G, 4: T. 
    '''
    rng = np.random.default_rng(seed)
    #1. Depths per haplotype
    DPh = depth_per_haplotype(rng, mean_depth, std_depth, gm.shape[1], ploidy)
    #2. Sample depths per SNP per haplotype
    DP  = rng.poisson(DPh, size=gm.shape)
    #3. Sample correct and error reads per SNP per haplotype (Rh)
    Rh  = np.array([rng.multinomial(dp, [1-e, e/3, e/3, e/3]).tolist() for dp in DP.reshape(-1)])
    Rh  = Rh.reshape(DP.shape[0], DP.shape[1], 4)
    #4. Reorganize anc and der alleles and join haplotypes to form individuals
    Rh_copy = np.copy(Rh)
    Rh[gm == 1, 1] = Rh_copy[gm == 1, 0]
    Rh[gm == 1, 0] = Rh_copy[gm == 1, 1]
    return Rh.reshape(Rh.shape[0], Rh.shape[1]//ploidy, ploidy, Rh.shape[2]).sum(axis = 2)

    
def allelereadcounts_to_GL(Rg, e = 0.05):
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

def allelereadcounts_to_pileup(allelereadcounts, filename = "tmp/reads.pileup"):
    with open(filename, "w") as out:
        first_line = True
        for i in range(allelereadcounts.shape[0]):
            line = "1\t"+str(i+1)+"\tN"
            for j in range(allelereadcounts.shape[1]):
                nreads = allelereadcounts[i, j, :].sum()
                line = line+"\t"+str(nreads)+"\t"
                if nreads:
                    for c, b in zip(allelereadcounts[i, j, :], ["A", "C", "G", "T"]):
                        line = line+c*b
                    line = line+"\t"+"."*nreads
                else:
                    line = line+"\t*\t*"
            out.write(line+"\n")

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

def allelereadcounts_to_vGL(allelereadcounts, e, ploidy = 2):
    GTxploidy    = np.array([list(x) for x in combinations_with_replacement([0, 1, 2, 3], ploidy)])
    AFxGTxploidy = np.array([(GTxploidy == 0).sum(axis = 1), (GTxploidy == 1).sum(axis = 1), (GTxploidy == 2).sum(axis = 1), (GTxploidy == 3).sum(axis = 1)])/ploidy
    
    GL_vec = np.multiply(-np.log(AFxGTxploidy*(1-e)+(1-AFxGTxploidy)*(e/3)), allelereadcounts.reshape(allelereadcounts.shape[0], allelereadcounts.shape[1], allelereadcounts.shape[2], 1)).sum(axis = 2)
    return GL_vec-GL_vec.min(axis = 2).reshape(GL_vec.shape[0], GL_vec.shape[1], 1)