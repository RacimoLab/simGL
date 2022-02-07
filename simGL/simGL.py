import numpy as np
from itertools import combinations_with_replacement
from itertools import combinations
from scipy.stats import binom

def incorporate_monomorphic(poly_gm, pos, start, end):
    '''
    Incorporates monomorphic sites in a polymorphic genotype matrix.

    Parameters
    ----------
    poly_gm : `numpy.ndarray` 
        Genotype matrix with size (polymorphic sites, haplotypic samples) in which 0 denotes reference allele
        and 1 denotes alternative allele.
    pos : `numpy.ndarray` 
        Genomic coordinates of the polymorphic sites with size (polymorphic sites, ) as integer or float values.
        The order of these values must be the same as the first dimetion of `poly_gm`.
    start : `int` or `float`
        Genomic start coordinate of the range for which monomorphic sites will be incorporated in the original
        `poly_gm` matrix. The value must be >= 0 <= min(pos).
    end : `int` or `float`
        Genomic end coordinate of the range for which monomorphic sites will be incorporated in the original
        `poly_gm` matrix. The value must be >= max(pos).
    
    Returns 
    -------
    gm : `numpy.ndarray`
        Genotype matrix with size (end-start, haplotypic samples) in which 0 denotes reference allele
        and 1 denotes alternative allele.
    '''
    if not (isinstance(poly_gm, np.ndarray) and len(poly_gm.shape) == 2 and ((gm == 0)+(gm == 1)).sum() == gm.size):
        raise TypeError('Incorrect `poly_gm` format: it has to be a numpy array with dimentions (polymorphic sites, haplotypic samples) ') 
    if not (isinstance(pos, np.ndarray) and len(pos.shape) == 1):
        raise TypeError('Incorrect `pos` format: it has to be a numpy array with dimentions (polymorphic sites, ) ')
    if not (isinstance(start, (int, float)) and start >= 0 and start <= min(pos)):
        raise TypeError('Incorrect `start` format: it has to be an integer value >=0 and <= min(pos) ') 
    if not (isinstance(end, (int, float)) and end >= max(pos)):
        raise TypeError('Incorrect `end` format: it has to be an integer value >= max(pos)') 
    if not (pos.shape[0] == poly_gm.shape[0]):
        raise TypeError('Incorrect `poly_gm` and/or `pos` format: They must have the same first dimention poly_gm.shape = (x, y) and  pos.shape = (x, )')
    gm = np.zeros((int(end)-int(start), poly_gm.shape[1]))
    gm[pos.astype(int)] = poly_gm
    return gm

def depth_per_haplotype(rng, mean_depth, std_depth, n_hap):
    if isinstance(mean_depth, np.ndarray):
        return mean_depth
    else:
        dp = rng.normal(loc = mean_depth, scale = std_depth, size=n_hap)
        dp[dp < 0.0] = -dp[dp == 0.0]
        return dp

def refalt_int_encoding(gm, ref, alt):
    refalt_str                    = np.array([ref, alt])
    refalt_int                    = np.zeros(refalt_str.shape, dtype=int)
    refalt_int[refalt_str == "C"] = 1
    refalt_int[refalt_str == "G"] = 2
    refalt_int[refalt_str == "T"] = 3
    return refalt_int[gm.reshape(-1), np.repeat(np.arange(gm.shape[0]), gm.shape[1])].reshape(gm.shape)

def sim_allelereadcounts(gm, mean_depth, e, ploidy, seed = None, std_depth = None, ref = None, alt = None):
    '''
    Simulates allele read counts from a genotype matrix. 
    
    Parameters
    ----------
    gm : `numpy.ndarray` 
        Genotype matrix with size (sites, haplotypic samples) in which 0 denotes reference allele
        and 1 denotes alternative allele.
    
    mean_depth : `int` or `float` or `numpy.ndarray`
        Read depth of the each haplotypic sample in `gm`. If a `int` or `float` value is inputed, the function
        will sample random values from a normal distribution with mean = `mean_depth` and std = `std_depth`.
        If a `numpy.ndarray` is inputed, the array must have size (haplotypic samples, ) and the order must
        be the same as the second dimention of `gm`.
    
    std_depth : `int` or `float`
        The standard deviation parameter of the normal distribution from which read depth values are randomly
        sampled for each haplotypic sample in `gm`. This value only needs to be provided if the `mean_depth`
        inputed is an `int` or a `float`.
    
    e : `int` or `float` 
        Sequencing error probability per base pair per site. The value must be between 0 and 1.
    
    ploidy : `int` 
        Number of haplotypic chromosomes per individual.
    
    ref : `numpy.ndarray`, optional
        Reference alleles list per site. The size of the array must be (sites, ) and the order has to 
        coincide with the first dimetion of `gm`. The values within the list must be strings {"A", "C", 
        "G", "T"}. If an `alt` list is inputed, a `ref` list must also be inputed. If no `ref` and `alt`
        are inputed, the `ref` allele is assumed to be "A" for all sites.
    
    alt : `numpy.ndarray`, optional
        Alternative alleles list per site. The size of the array must be (sites, ) and the order has to 
        coincide with the first dimetion of `gm`. The values within the list must be strings {"A", "C", 
        "G", "T"}. If a `ref` list is inputed, an `alt` list must also be inputed. If no `ref` and `alt`
        are inputed, the `alt` allele is assumed to be "C" for all sites.

    seed : `int`, optional
        Starting point in generating random numbers.
    
    Returns 
    -------
    arc : `numpy.ndarray`
        Allele read counts per site per individual. The dimentions of the array are (sites, individuals, alleles). 
        The third dimention of the array has size = 4, which corresponds to the four possible alleles: 0 = "A", 
        1 = "C", 2 = "G" and 3 = "T".
    
    Notes
    -----
    - The read depth indicated in `mean_depth` is per haplotypic sample, i.e. if the user intends to simulate a 
      depth of 30 reads per site per individual, and individuals are diploid (`ploidy` = 2), the `mean_depth` 
      must be 15. 
    - If monomorphic sites are included, the `alt` values corresponding to those sites are not taken into account, 
      but they must be still indicated.
    '''
    #Checks
    if not (isinstance(gm, np.ndarray) and len(gm.shape) == 2 and ((gm == 0)+(gm == 1)).sum() == gm.size):
        raise TypeError('Incorrect gm` format: it has to be a numpy array with dimentions (sites, haplotypic samples) with integer values 1 and 0')
    if not ((isinstance(mean_depth, np.ndarray) and len(mean_depth.shape) == 1 and mean_depth.shape[0] == gm.shape[1] and (mean_depth > 0).sum() == mean_depth.size) or (isinstance(mean_depth, (int, float)) and mean_depth > 0.0)):
        raise TypeError('Incorrect `mean_depth` format: it has to be either i) numpy.array with dimentions (haplotypic samples, ) with values > 0 or ii) integer or float value > 0')
    if not ((isinstance(mean_depth, np.ndarray)) or (isinstance(std_depth, (int, float)) and std_depth >= 0.0)):
        raise TypeError('Incorrect `std_depth` format: it has to be an integer or float value > 0 if mean_depth is a integer or float value and not a numpy array')
    if not (isinstance(e, (int, float)) and e >= 0.0 and e <= 1.0) :
        raise TypeError('Incorrect `e` format: it has to be a float value >= 0 and <= 1')
    if not (isinstance(ploidy, int) and ploidy > 0 and gm.shape[1]%ploidy == 0) :
        raise TypeError('Incorrect `ploidy` format: it has to be an integer value > 0 the second dimention of `gm` (haplotypic samples) must be divisible by ploidy')
    if ref is None and alt is None:
        ref = np.full(gm.shape[0], "A")
        alt = np.full(gm.shape[0], "C")
    elif not (isinstance(ref, np.ndarray) and isinstance(alt, np.ndarray) and len(ref.shape) == 1 and len(alt.shape) == 1 and ref.shape == alt.shape and ref.size == gm.shape[0] and
              ((ref == "A") + (ref == "C") + (ref == "G") + (ref == "T")).sum() == ref.size and ((alt == "A") + (alt == "C") + (alt == "G") + (alt == "T")).sum() == alt.size):
        raise TypeError('Incorrect `ref` and/or `alt` format: they both have to be a numpy array with dimentions (sites, ) with string "A", "C", "G", "T" values')
    #Variables
    err = np.array([[1-e, e/3, e/3, e/3], [e/3, 1-e, e/3, e/3], [e/3, e/3, 1-e, e/3], [e/3, e/3, e/3, 1-e]])
    rng = np.random.default_rng(seed)
    #1. Depths (DP) per haplotype (h)
    DPh = depth_per_haplotype(rng, mean_depth, std_depth, gm.shape[1])
    #2. Sample depths (DP) per site per haplotype
    DP  = rng.poisson(DPh, size=gm.shape)
    #3. Sample correct and error reads per SNP per haplotype (Rh)
    #3.1. Convert anc = 0/der = 1 encoded gm into "A" = 0, "C" = 1, "G" = 3, "T" = 4 basepair (bp) encoded gm 
    gmbp = refalt_int_encoding(gm, ref, alt)
    #3.2. Simulate allele read counts (ARC) per haplotype (h) per site (s)
    arc  = rng.multinomial(DP, err[gmbp])
    #4. Add n haplotype read allele counts (n = ploidy) to obtain read allele counts per genotype
    return arc.reshape(arc.shape[0], arc.shape[1]//ploidy, ploidy, arc.shape[2]).sum(axis = 2)

def get_GTxploidy(ploidy):
    return np.array([list(x) for x in combinations_with_replacement([0, 1, 2, 3], ploidy)])

def allelereadcounts_to_GL(arc, e, ploidy):
    '''
    Computes genotype likelihoods from allele read counts per site per individual. 
    
    Parameters
    ----------
    arc : `numpy.ndarray`
        Allele read counts per site per individual. The dimentions of the array are (sites, individuals, alleles). 
        The third dimention of the array has size = 4, which corresponds to the four possible alleles: 0 = "A", 
        1 = "C", 2 = "G" and 3 = "T".
    
    e : `float` 
        Sequencing error probability per base pair per site. The value must be between 0 and 1.

    ploidy : `int` 
        Number of haplotypic chromosomes per individual.  

    Returns 
    -------

    GL : `numpy.ndarray`
        Normalized genotype likelihoods per site per individual. The dimentions of the array are (sites, individuals, genotypes). 
        The third dimention of the array corresponds to the combinations with replacement of all 4 possible alleles 
        {"A", "C", "G", "T"} (i.e., for a diploid, there are 10 possible genotypes and the combination order is "AA", "AC",
        "AG", "AT", "CC", "CG", ..., "TT"). 

    References
    ----------
    1) McKenna A, Hanna M, Banks E, Sivachenko A, Cibulskis K, Kernytsky A, Garimella K, Altshuler D, Gabriel S, Daly M, DePristo MA (2010). The Genome Analysis Toolkit: a MapReduce framework for analyzing next-generation DNA sequencing data. Genome Res. 20:1297-303.
    2) Thorfinn Sand Korneliussen, Anders Albrechtsen, Rasmus Nielsen. ANGSD: Analysis of Next Generation Sequencing Data. BMC Bioinform. 2014 Nov;15,356.
    '''
    if not (isinstance(arc, np.ndarray) and len(arc.shape) == 3 and arc.shape[2] == 4):
        raise TypeError('Incorrect `arc` format: it has to be a numpy array with dimentions (sites, individuals, alleles) and the third dimention must be of size = 4')
    if not (isinstance(e, float) and e >= 0 and e <= 1) :
        raise TypeError('Incorrect `e` format: it has to be a float value >= 0 and <= 1')
    if not (isinstance(ploidy, int) and ploidy > 0) :
        raise TypeError('Incorrect `ploidy` format: it has to be an integer value > 0')

    GTxploidy    = get_GTxploidy(ploidy)
    AFxGTxploidy = np.array([(GTxploidy == 0).sum(axis = 1), (GTxploidy == 1).sum(axis = 1), (GTxploidy == 2).sum(axis = 1), (GTxploidy == 3).sum(axis = 1)])/ploidy
    
    GL = np.multiply(-np.log(AFxGTxploidy*(1-e)+(1-AFxGTxploidy)*(e/3)), arc.reshape(arc.shape[0], arc.shape[1], arc.shape[2], 1)).sum(axis = 2)
    return GL-GL.min(axis = 2).reshape(GL.shape[0], GL.shape[1], 1)
    
def get_pGTxMm(ploidy):
    GTxploidy    = np.array([list(x) for x in combinations_with_replacement([0, 1, 2, 3], ploidy)])
    Mmxploidy    = np.array([list(x) for x in combinations([0, 1, 2, 3], 2)])
    pGTxMm = []
    #For every genotype (GT)
    for i in range(GTxploidy.shape[0]):
        pGTxMm_tmp = []
        #For every combination of major (M) and minor (m) alleles (M and m can't be the same allele and there can be only two)
        for j in range(Mmxploidy.shape[0]):
            #All alleles in GT are either M or m
            all_GT_in_Mm = (np.isin(GTxploidy[i],  Mmxploidy[j]).sum() == ploidy)*1
            #Probability of the genotype given M and m only possible alleles
            p_GT = binom.pmf((GTxploidy[i] == Mmxploidy[j, 0]).sum(), ploidy, 0.5)
            pGTxMm_tmp.append( p_GT * all_GT_in_Mm )
        pGTxMm.append(np.array(pGTxMm_tmp))
    return np.array(pGTxMm)

def GL_to_Mm(GL, ploidy):
    '''
    Computes maximum (M) and minimum (m) frequency alleles in the population from genotype likelihoods. 
    
    Parameters
    ----------
    GL : `numpy.ndarray`
        Normalized genotype likelihoods per site per individual. The dimentions of the array is (sites, individuals, genotypes). 
        The third dimention of the array corresponds to the combinations with replacement of all 4 possible alleles 
        {"A", "C", "G", "T"} (i.e., for a diploid, there are 10 possible genotypes and the combination order is "AA", "AC",
        "AG", "AT", "CC", "CG", ..., "TT"). 

    ploidy : `int` 
        Number of haplotypic chromosomes per individual. 

    Returns 
    -------
    `numpy.ndarray`
        Maximum and minimum alleles per site. The dimentions of the array is (sites, ) and the values per site is an integer 
        encoding the pair of M and m: 0 = "AC", 1 = "AG", 2 = "AT", 3 = "CG", 4 = "CT", 5 = "GT".
    
    References
    ----------
    1) Line Skotte, Thorfinn Sand Korneliussen, Anders Albrechtsen. Association testing for next-generation sequencing data using score statistics. Genet Epidemiol. 2012 Jul;36(5):430-7.
    2) Thorfinn Sand Korneliussen, Anders Albrechtsen, Rasmus Nielsen. ANGSD: Analysis of Next Generation Sequencing Data. BMC Bioinform. 2014 Nov;15,356.
    '''
    if not (isinstance(GL, np.ndarray) and len(GL.shape) == 3):
        raise TypeError('Incorrect `GL` format: it has to be a numpy array with dimentions (sites, individuals, genotypes)')
    if not (isinstance(ploidy, int) and ploidy > 0 and gm.shape[1]%ploidy == 0):
        raise TypeError('Incorrect `ploidy` format: it has to be an integer value > 0 the second dimention of `gm` (haplotypic samples) must be divisible by ploidy')
    if get_GTxploidy(ploidy).size != GL.shape[2]:
        raise TypeError('Incorrect `ploidy` format or `GL` shape: the third dimention of `GL` {} does not correspond with `ploidy` value {}'.format(GL.shape[2], get_GTxploidy(ploidy).size))

    pGTxMm = get_pGTxMm(ploidy)
    return np.argmin((GL.reshape(GL.shape[0], GL.shape[1], GL.shape[2], 1) * pGTxMm.reshape(1, 1, pGTxMm.shape[0], pGTxMm.shape[1])).sum(axis = 2).prod(axis = 1), axis = 1)

def allelereadcounts_to_pileup(arc, output):
    '''
    Writes an allele read counts in a file in pileup format.

    Parameters
    ----------
    arc : `numpy.ndarray`
        Allele read counts per site per individual. The dimentions of the array are (sites, individuals, alleles). 
        The third dimention of the array has size = 4, which corresponds to the four possible alleles: 0 = "A", 
        1 = "C", 2 = "G" and 3 = "T".
    
    output : `str`
        Output file name.

    Returns 
    -------
    None
    '''
    if not (isinstance(arc, np.ndarray) and len(arc.shape) == 3 and arc.shape[2] == 4):
        raise TypeError('Incorrect `arc` format: it has to be a numpy array with dimentions (sites, individuals, alleles) and the third dimention must be of size = 4')
    if not (isinstance(output, str)):
        raise TypeError('Incorrect `output` format: it has to be a string with the path where the output is written')
    with open(output, "w") as out:
        for i in range(arc.shape[0]):
            line = "1\t"+str(i+1)+"\tN"
            for j in range(arc.shape[1]):
                nreads = arc[i, j, :].sum()
                line = line+"\t"+str(nreads)+"\t"
                if nreads:
                    for c, b in zip(arc[i, j, :], ["A", "C", "G", "T"]):
                        line = line+c*b
                    line = line+"\t"+"."*nreads
                else:
                    line = line+"\t*\t*"
            out.write(line+"\n")