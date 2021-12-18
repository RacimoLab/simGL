import numpy as np
import tskit

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

def depth_per_haplotype(rng, mean_depth, std_depth, n_ind):
    if type(mean_depth) == int or type(mean_depth) == float:
        if type(std_depth) == int or type(std_depth) == float:
            DPh = []
            while len(DPh) < n_ind:
                dp = rng.normal(loc = mean_depth/2, scale = std_depth, size=1)[0]
                if dp > 0:
                    DPh.append(dp)
            return DPh
    elif type(mean_depth) == np.ndarray and len(mean_depth.shape) == 1 and mean_depth.shape[0] == n_ind and (mean_depth > 0).sum() == n_ind:
        return mean_depth
    else:
        sys.exit("Incorrect mean_depth format")
        
        
def sim_allelereadcounts(data, mean_depth = 30., std_depth = 5., e = 0.05, seed = 1234):
    '''
    Def:
        Function to simulate read counts for alleles given a tree sequence data from diploid simulated individuals (2samples = ind) 
        or genotype matrix and extra information for the haplotype samples.
    Input:
        - data       : Two inputs are possible:
                            + Tree sequence data from tskit package from which the genotype matrix is going to be extracted
                            + Genotype matrix in numpy format with shape (SNPs, samples). It is assumed that the array is sorted
                              according to a individual order such that consecutive columns (e.g., data[:, 0] and data[:, 1]) 
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
        - seed       : integer from which the numpy rng will be drawn. 
    Output:
        - Rg         : numpy array with dimentions (SNP, individual, alleles) so that each value corresponds to the number of
                       reads with a particular allele for a SNP position and a diploid individual. The index of the 3rd dimention
                       corresponds to 1 : A and ancestral allele, 2 : C and derived allele, 3 : G, 4: T. 
    '''
    rng = np.random.default_rng(seed)
    M   = extract_genotype_matrix(data)
    #1. Depths per haplotype
    DPh = depth_per_haplotype(rng, mean_depth, std_depth, M.shape[1])
    #2. Sample depths per SNP per haplotype
    DP  = rng.poisson(DPh, size=M.shape)
    #3. Sample correct and error reads per SNP per haplotype (Rh)
    Rh  = np.array([rng.multinomial(dp, [1-e, e/3, e/3, e/3]).tolist() for dp in DP.reshape(-1)])
    Rh  = Rh.reshape(DP.shape[0], DP.shape[1], 4)
    #4. Reorganize anc and der alleles and join haplotypes to form individuals
    Rh_copy = np.copy(Rh)
    Rh[M == 1, 1] = Rh_copy[M == 1, 0]
    Rh[M == 1, 0] = Rh_copy[M == 1, 1]
    return Rh.reshape(Rh.shape[0], Rh.shape[1]//2, 2, Rh.shape[2]).sum(axis = 2)

    
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

    GL = np.array(GL)
    return GL - GL.min(axis = 0)

