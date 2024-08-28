import numpy as np
from itertools import combinations_with_replacement
from itertools import combinations
from scipy.stats import binom


def incorporate_monomorphic(gm, pos, start, end):
    '''
    Incorporates monomorphic sites in a polymorphic genotype matrix.

    Parameters
    ----------
    gm  : `numpy.ndarray`, dtype `int`
        Genotype matrix with size (polymorphic sites, haplotypic samples) in which 0 denotes reference allele
        and 1 denotes alternative allele.
    pos : `numpy.ndarray`, dtype `int` 
        Genomic coordinates of the polymorphic sites with size (polymorphic sites, ) as integer values >= 0.
        The values must be sorted and the order of these values must be the same as the first dimetion of `gm`.
    start : `int`
        Genomic start coordinate of the range for which monomorphic sites will be incorporated in the original
        `gm` matrix. The value must be >= 0 <= min(pos).
    end : `int`
        Genomic end coordinate of the range for which monomorphic sites will be incorporated in the original
        `gm` matrix. The value must be >= max(pos).
    
    Returns 
    -------
    gm2 : `numpy.ndarray`, dtype `int`
        Genotype matrix with size (end-start, haplotypic samples) in which 0 denotes reference allele
        and 1 denotes alternative allele.
    '''
    assert check_gm(gm) and check_pos(gm, pos) and check_start(pos, start) and check_end(pos, end)
    gm2 = np.zeros((int(end)-int(start), gm.shape[1]))
    gm2[pos.astype(int)] = gm
    return gm2

def depth_per_haplotype(rng, mean_depth, std_depth, n_hap, ploidy):
    if isinstance(mean_depth, np.ndarray):
        return mean_depth
    else:
        shape = np.power(mean_depth, 2)/np.power(std_depth, 2)
        scale = np.power(std_depth, 2)/mean_depth
        return rng.gamma(shape = shape, scale=scale, size=n_hap)

def refalt_int_encoding(gm, ref, alt):
    refalt_str                    = np.array([ref, alt])
    refalt_int                    = np.zeros(refalt_str.shape, dtype=int)
    refalt_int[refalt_str == "C"] = 1
    refalt_int[refalt_str == "G"] = 2
    refalt_int[refalt_str == "T"] = 3
    return refalt_int[gm.reshape(-1), np.repeat(np.arange(gm.shape[0]), gm.shape[1])].reshape(gm.shape)

def linked_depth(rng, DPh, read_length, start, end, pos):
    '''
    Simulates reads in a genomic region to compute the depth per position. The region
    simulatied should contain the loci for which the user is interested (i.e, the start
    and end coordinates of the region simulated must be lower and higher than the coordinates
    of the loci).

    In this function, the end of the sequence is connected to the start of the sequence, so reads simulated 
    that extend over the end of the sequence map at the beggining of it. For example, a simulated region of 
    10,000 bp (start position 0, end position 9,999), if a 100 bp read maps at position 9,950, the first 50 
    bp will map at the end of the simulated sequence (9,950 to 9,999) and the last 50 bp will map at the 
    beggining of the simulated sequence (0 to 49).This creates an artifact that the coverage of sites at the 
    end and start of the sequence are correlated. This is done to ensure that the coverage simulated for the 
    simulated sequence is the same as the inputed mean coverage.
    
    Parameters
    ----------
    rng : `numpy.random._generator.Generator` 
        random number generation numpy object
    DPh : `numpy.ndarray`, dtype `int` or `float`
        Numpy array with the depth per haplotype
    read_length : `int`
        Read length in base pair units
    start : `int`
        Starting position of the region simulated
    end : `int`
        Ending position of the region simulated
    pos : `numpy.ndarray`, dtype `int`
        Genomic coordinates of the loci of interest
    
    Returns 
    -------
    DP : `numpy.ndarray`, dtype `int`
        Depth per site per haplotype corresponding to
        all the loci of interest
    '''
    DP = []
    reg_length = end-start
    read_n     = ((DPh*reg_length)/read_length).astype("int")
    for r in read_n:
        dp = np.zeros((reg_length,), dtype=int)
        for start in rng.integers(low=0, high=reg_length-1, size=r):
            end = start+read_length
            dp[start:end] += 1
            dp[0:max(0, end-reg_length)] += 1
        DP.append(dp.tolist())
    return np.array(DP).T[pos-start]

def independent_depth(rng, DPh, size):
    '''
    Returns depth per position per haplotype (size[0], size[1]) drawn from the "rng" from a Poisson 
    distribution with a lambda value "DPh" per haplotype
    '''
    return rng.poisson(DPh, size=size)

def sim_allelereadcounts(gm, mean_depth, e, ploidy, seed = None, std_depth = None, ref = None, alt = None, depth_type = "independent", read_length = None, start = None, end = None, pos = None):
    '''
    Simulates allele read counts from a genotype matrix. 
    
    Parameters
    ----------
    gm : `numpy.ndarray`, dtype `int`
        Genotype matrix with size (sites, haplotypic samples) in which 0 denotes reference allele
        and 1 denotes alternative allele.
    
    mean_depth : `int` or `float` or `numpy.ndarray`, dtype `int` or `float`
        Read depth of the each haplotypic sample in `gm`. If a `int` or `float` value is inputed, the function
        will sample random values from a gamma distribution with mean = `mean_depth` and std = `std_depth`. 
        If the distribution is far from 0, the gamma distribution is like a normal distribution with the
        indicated mean and standard deviation. 
        If a `numpy.ndarray` is inputed, the array must have size (haplotypic samples, ) and the order must
        be the same as the second dimention of `gm`.
    
    std_depth : `int` or `float`
        The standard deviation parameter of the values generated by the gamma distribution from which read depth 
        values are randomly sampled for each haplotypic sample in `gm`. This value only needs to be provided if 
        the `mean_depth` inputed is an `int` or a `float`.
    
    e : `int` or `float` 
        Sequencing error probability per base pair per site. The value must be between 0 and 1.
    
    ploidy : `int` 
        Number of haplotypic chromosomes per individual.
    
    ref : `numpy.ndarray`, dtype `str`, optional
        Reference alleles list per site. The size of the array must be (sites, ) and the order has to 
        coincide with the first dimetion of `gm`. The values within the list must be strings {"A", "C", 
        "G", "T"}. If an `alt` list is inputed, a `ref` list must also be inputed. If no `ref` and `alt`
        are inputed, the `ref` allele is assumed to be "A" for all sites.
    
    alt : `numpy.ndarray`, dtype `str`, optional
        Alternative alleles list per site. The size of the array must be (sites, ) and the order has to 
        coincide with the first dimetion of `gm`. The values within the list must be strings {"A", "C", 
        "G", "T"}. If a `ref` list is inputed, an `alt` list must also be inputed. If no `ref` and `alt`
        are inputed, the `alt` allele is assumed to be "C" for all sites.

    depth_type : `str` {`independent`, `linked`}, default = independent
        Method to simulate coverage per loci.

        - independent
            The coverage of every loci in `gm` will be independently simulated from a poisson distribution 
            with lambda value equal to the mean depth for each haplotype.

        - linked
            Coverage is obtained by simulating a number of reads that satisfies the inputed average coverage,
            the read length and the length of the sequence, placing them randombly along the genomic region
            simulated. If this option is enabled, the arguments read_length, start, end and pos must
            also be used. Read more about this simulation mode in the notes section below.

    read_length : `int`, required with `depth_type` = `linked`
        Length of the reads simulated in base pair units if linked coverage is enabled. The value must be > 0.

    start : `int`, required with `depth_type` = `linked`
        Genomic start coordinate of the of the region simulated if linked coverage is enabled. The value must 
        be >= 0 <= min(`pos`).

    end : `int`, required with `depth_type` = `linked`
        Genomic end coordinate of the of the region simulated if linked coverage is enabled. The value must 
        be >= max(`pos`). 
    
    pos : `numpy.ndarray`, dtype `int`, required with depth_type = linked
        Genomic coordinates of the polymorphic sites with size (polymorphic sites, ) as integer values >= 0.
        The values must be sorted and the order of these values must be the same as the first dimetion of `gm`.

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
      but they must be in the array.
    - In the linked simulation mode the end of the sequence is connected to the start of the sequence, so reads 
      simulated that extend over the end of the sequence map at the beggining of it. For example, a simulated region of 
      10,000 bp (start position 0, end position 9,999), if a 100 bp read maps at position 9,950, the first 50 bp will 
      map at the end of the simulated sequence (9,950 to 9,999) and the last 50 bp will map at the beggining of the 
      simulated sequence (0 to 49).This creates an artifact that the coverage of sites at the end and start of the 
      sequence are correlated. This is done to ensure that the coverage simulated for the simulated sequence is the same 
      as the inputed mean coverage.
    '''
    #Checks
    assert check_gm(gm)
    if ref is None and alt is None:
        ref = np.full(gm.shape[0], "A")
        alt = np.full(gm.shape[0], "C")
    assert check_mean_depth(gm, mean_depth) and check_std_depth(mean_depth, std_depth) and check_e(e) and check_ploidy(ploidy) and check_gm_ploidy(gm, ploidy) and check_ref_alt(gm, ref, alt) and check_depth_type(depth_type)
    #Variables
    err = np.array([[1-e, e/3, e/3, e/3], [e/3, 1-e, e/3, e/3], [e/3, e/3, 1-e, e/3], [e/3, e/3, e/3, 1-e]])
    rng = np.random.default_rng(seed)
    #1. Depths (DP) per haplotype (h)
    DPh = depth_per_haplotype(rng, mean_depth, std_depth, gm.shape[1], ploidy)
    #2. Sample depths (DP) per site per haplotype
    if depth_type == "independent":
        DP  = independent_depth(rng, DPh, gm.shape)
    elif depth_type == "linked":
        assert check_positive_nonzero_integer(read_length, "read_length") and check_pos(gm, pos) and check_start(pos, start) and check_end(pos, end) and check_reads_smaller_than_genomic_region(read_length, start, end)
        DP  = linked_depth(rng, DPh, read_length, start, end, pos)
    assert DP.shape == gm.shape # TODO: Not here, let's put it as a test
    #3. Sample correct and error reads per SNP per haplotype (Rh)
    #3.1. Convert anc = 0/der = 1 encoded gm into "A" = 0, "C" = 1, "G" = 3, "T" = 4 basepair (bp) encoded gm 
    gmbp = refalt_int_encoding(gm, ref, alt)
    #3.2. Simulate allele read counts (ARC) per haplotype (h) per site (s)
    arc  = rng.multinomial(DP, err[gmbp])
    #4. Add n haplotype read allele counts (n = ploidy) to obtain read allele counts per genotype
    return arc.reshape(arc.shape[0], arc.shape[1]//ploidy, ploidy, arc.shape[2]).sum(axis = 2)

def array_combinations_with_replacement(elements, positions):
    return np.array([list(x) for x in combinations_with_replacement(elements, positions)])

def array_combinations(elements, positions):
    return np.array([list(x) for x in combinations(elements, positions)])

def normalize_GL(GL):
    return GL-GL.min(axis = 2).reshape(GL.shape[0], GL.shape[1], 1)

def allelereadcounts_to_GL(arc, e, ploidy, normalized = True):
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

    normalized :  `bool`
        Boolean variable that determines if the output of the function should be normalized. Normalization consists 
        on substracting the value of the most likely genotype in every genotype per loci and individual.

    Returns 
    -------

    GL : `numpy.ndarray`
        Negative log genotype likelihoods per site per individual. The dimentions of the array are (sites, individuals, genotypes). 
        The third dimention of the array corresponds to the combinations with replacement of all 4 possible alleles 
        {"A", "C", "G", "T"} (i.e., for a diploid, there are 10 possible genotypes and the combination order is "AA", "AC",
        "AG", "AT", "CC", "CG", ..., "TT"). 

    References
    ----------
    1) McKenna A, Hanna M, Banks E, Sivachenko A, Cibulskis K, Kernytsky A, Garimella K, Altshuler D, Gabriel S, Daly M, DePristo MA (2010). The Genome Analysis Toolkit: a MapReduce framework for analyzing next-generation DNA sequencing data. Genome Res. 20:1297-303.
    2) Thorfinn Sand Korneliussen, Anders Albrechtsen, Rasmus Nielsen. ANGSD: Analysis of Next Generation Sequencing Data. BMC Bioinform. 2014 Nov;15,356.
    '''
    assert check_arc(arc) and check_e(e) and check_ploidy(ploidy)
    
    GTxploidy    = array_combinations_with_replacement([0, 1, 2, 3], ploidy)
    AFxGTxploidy = np.array([(GTxploidy == 0).sum(axis = 1), (GTxploidy == 1).sum(axis = 1), (GTxploidy == 2).sum(axis = 1), (GTxploidy == 3).sum(axis = 1)])/ploidy
    
    GL = np.multiply(-np.log(AFxGTxploidy*(1-e)+(1-AFxGTxploidy)*(e/3)), arc.reshape(arc.shape[0], arc.shape[1], arc.shape[2], 1)).sum(axis = 2)
    if normalized:
        return normalize_GL(GL)
    return GL
    
def get_pGTxMm(ploidy):
    GTxploidy    = array_combinations_with_replacement([0, 1, 2, 3], ploidy)
    Mmxploidy    = array_combinations([0, 1, 2, 3], ploidy)
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

def get_Mmindex(GL, ploidy):
    pGTxMm    = get_pGTxMm(ploidy)
    return np.argmin((GL.reshape(GL.shape[0], GL.shape[1], GL.shape[2], 1) * pGTxMm.reshape(1, 1, pGTxMm.shape[0], pGTxMm.shape[1])).sum(axis = 2).prod(axis = 1), axis = 1)

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
        Maximum and minimum alleles per site. The dimentions of the array is (sites, ploidy) and the values per site is an integer 
        encoding the Maximum and minimum alleles: 0 = "A", 1 = "C", 2 = "G", 3 = "T". The index of the alleles are sorted by
        order, not corresponding to the maximum and minimum alleles.
    
    References
    ----------
    1) Line Skotte, Thorfinn Sand Korneliussen, Anders Albrechtsen. Association testing for next-generation sequencing data using score statistics. Genet Epidemiol. 2012 Jul;36(5):430-7.
    2) Thorfinn Sand Korneliussen, Anders Albrechtsen, Rasmus Nielsen. ANGSD: Analysis of Next Generation Sequencing Data. BMC Bioinform. 2014 Nov;15,356.
    '''
    # TODO: when there are too many individuals, the numeric operation is not sable.
    assert check_ploidy(ploidy) and check_GL(GL, ploidy)
    Mmxploidy = array_combinations([0, 1, 2, 3], ploidy)
    Mmindex   = get_Mmindex(GL+1, ploidy)
    return Mmxploidy[Mmindex]

def sorted_unsorted_allele_combination_correspondance(alleles_per_site, ploidy):
    unsorted_rows = np.all(np.diff(alleles_per_site) < 0, axis = 1) #check which rows (loci) have alleles which are not in order, and thus, are in the wrong order in GL_subset
    correspondance = {}
    for pattern in np.unique(alleles_per_site[unsorted_rows], axis = 0): # Make a dictionary that will for every pattern of alleles make the corresponance of genotype positions according to the order
        sorted_combination   = [list(x) for x in combinations_with_replacement(np.sort(np.array(pattern)).tolist(), ploidy)]
        unsorted_combination = [list(x) for x in combinations_with_replacement(np.array(pattern).tolist(), ploidy)]
        correspondance[tuple(pattern.tolist())] = [j for i in range(len(unsorted_combination)) for j in range(len(sorted_combination)) if np.all(np.isin(sorted_combination[j], unsorted_combination[i])) and np.all(np.isin(unsorted_combination[i], sorted_combination[j]))]
    return correspondance, unsorted_rows

def subset_GL(GL, alleles_per_site, ploidy):
    '''
    Extracts the relevant genotype likelihoods form a `GL` array according to the requested alleles. 

    Parameters
    ----------
    GL : `numpy.ndarray`
        Normalized genotype likelihoods per site per individual. The dimentions of the array is (sites, individuals, genotypes). 
        The third dimention of the array corresponds to the combinations with replacement of all 4 possible alleles 
        {"A", "C", "G", "T"} (i.e., for a diploid, there are 10 possible genotypes and the combination order is "AA", "AC",
        "AG", "AT", "CC", "CG", ..., "TT"). 

    alleles_per_site : `numpy.ndarray`
        Index of the alleles of interest per site. All sites must contain the same number of queried alleles. The index of the
        alleles is : 0 = "A", 1 = "C", 2 = "G" and 3 = "T". An example of this array is the output of `GL_to_Mm` in which each
        locus contains the allele index of the maximum and minimum frequent alleles in the dataset according to `GL`.

    ploidy : `int` 
        Number of haplotypic chromosomes per individual. 

    Returns 
    -------
    `numpy.ndarray`
        Subset of the inputed `GL` matrix with the GT likelihoods corresponding to the queried alleles.
    '''
    assert check_ploidy(ploidy) and check_GL(GL, ploidy) and check_alleles_per_site(GL, alleles_per_site)
    GTxploidy = array_combinations_with_replacement([0, 1, 2, 3], ploidy)

    n_loci, n_ind, n_GL = GL.shape
    n_genotypes = array_combinations_with_replacement(np.arange(alleles_per_site.shape[1]), ploidy).shape[0]

    dim1 = np.repeat(np.arange(n_loci), n_genotypes*n_ind)
    dim2 = np.repeat(np.tile(np.arange(n_ind), n_loci), n_genotypes)

    GTidxxn_loci = np.tile(np.arange(n_GL), n_loci).reshape(n_loci, n_GL) #Construct a dummy matrix with dimetions (n_loci, n_GL) that will have the index of each GT
    GTidxxn_loci_bool = ((alleles_per_site.reshape(n_loci, ploidy, 1, 1) == GTxploidy.T.reshape(1, 1, ploidy, n_GL)).sum(axis = 2).sum(axis = 1) == ploidy) # boolean matrix with dimentions (n_loci, n_GL) that encodes which GT positions correspond to the Mm alleles
    dim3 = np.tile(GTidxxn_loci[GTidxxn_loci_bool].reshape(n_loci, -1), n_ind).reshape(-1) #Array with the GT index positions to retrieve, repeated (tiled) as many times as n_ind

    GL_subset = GL[dim1, dim2, dim3].reshape(n_loci, n_ind, n_genotypes)

    #The GL subset gives the GL sorted according to allele index. So, even though in position x the allels inputed were [2, 1], the genotypes correspond to [[1,1], [1,2], [2,2]] now I need a way of reversing the order
    
    correspondance, unsorted_rows = sorted_unsorted_allele_combination_correspondance(alleles_per_site, ploidy) #Make a dictionary that for every pattern of alleles unsorted, tells which index the every genotype is found
    n_unsorted_rows = unsorted_rows.sum()

    dim1 = np.repeat(np.where(unsorted_rows)[0], n_ind*n_genotypes)
    dim2 = np.tile(np.repeat(np.arange(n_ind), n_genotypes), n_unsorted_rows)
    dim3 = np.tile(np.array([correspondance[tuple(x.tolist())] for x in alleles_per_site[unsorted_rows]]), n_ind).reshape(-1)

    GL_subset[unsorted_rows] = GL_subset[dim1, dim2, dim3].reshape(n_unsorted_rows, n_ind, n_genotypes)
    return GL_subset

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

# Functions to check input formatting
def check_gm(gm):
    if not (isinstance(gm, np.ndarray) and len(gm.shape) == 2 and ((gm == 0)+(gm == 1)).sum() == gm.size):
        raise TypeError('Incorrect gm format: it has to be a numpy array with dimentions (sites, haplotypic samples) with integer values 1 and 0')
    return True

def check_mean_depth(gm, mean_depth):
    if not ((isinstance(mean_depth, np.ndarray) and len(mean_depth.shape) == 1 and mean_depth.shape[0] == gm.shape[1] and (mean_depth > 0).sum() == mean_depth.size) or (isinstance(mean_depth, (int, float)) and mean_depth > 0.0)):
        raise TypeError('Incorrect mean_depth format: it has to be either i) numpy.array with dimentions (haplotypic samples, ) with values > 0 or ii) integer or float value > 0')
    return True

def check_std_depth(mean_depth, std_depth):
    if not ((isinstance(mean_depth, np.ndarray)) or (isinstance(std_depth, (int, float)) and std_depth >= 0.0)):
        raise TypeError('Incorrect std_depth format: it has to be an integer or float value > 0 if mean_depth is a integer or float value and not a numpy array')
    return True

def check_e(e):
    if not (isinstance(e, (int, float)) and e >= 0.0 and e <= 1.0) :
        raise TypeError('Incorrect e format: it has to be a float value >= 0 and <= 1')
    return True

def check_ploidy(ploidy):
    if not (isinstance(ploidy, int) and ploidy > 0) :
        raise TypeError('Incorrect ploidy format: it has to be an integer value > 0')
    return True

def check_gm_ploidy(gm, ploidy):
    if not (gm.shape[1]%ploidy == 0) :
        raise TypeError('Incorrect ploidy and/or gm format: the second dimention of gm (haplotypic samples) must be divisible by ploidy')
    return True

def check_depth_type(depth_type):
    if not isinstance(depth_type, str) or depth_type not in ["independent", "linked"]:
        raise TypeError('Incorrect depth_type format: it has to be a string, either "independent" or "linked"')
    return True

def check_positive_nonzero_integer(value, name):
    if not isinstance(value, int) or value <= 0:
        raise TypeError('Incorrect {} format: it has to be a integer value > 0'.format(name))
    return True

def check_ref_alt(gm, ref, alt):
    if not (isinstance(ref, np.ndarray) and isinstance(alt, np.ndarray) and len(ref.shape) == 1 and len(alt.shape) == 1 and ref.shape == alt.shape and ref.size == gm.shape[0] and
              ((ref == "A") + (ref == "C") + (ref == "G") + (ref == "T")).sum() == ref.size and ((alt == "A") + (alt == "C") + (alt == "G") + (alt == "T")).sum() == alt.size):
        raise TypeError('Incorrect ref and/or alt format: they both have to be a numpy array with dimentions (sites, ) with string "A", "C", "G", "T" values')
    return True

def check_pos(gm, pos):
    if not (isinstance(pos, np.ndarray) and len(pos.shape) == 1 and (pos >= 0).sum() == pos.size and pos.shape[0] == gm.shape[0] and np.issubdtype((pos).dtype, np.integer) and (pos[:-1] >= pos[1:]).sum() == 0):        
        raise TypeError('Incorrect pos format: it has to be a sorted numpy array with dimentions (polymorphic sites, ) corresponding to the genomic coordinates of loci')
    return True

def check_start(pos, start):
    if not (isinstance(start, int) and start >= 0 and start <= pos[0]):
        raise TypeError('Incorrect start format: it has to be an integer value >=0 and <= pos[0] (minimum position value) ')
    return True

def check_end(pos, end):
    if not (isinstance(end, int) and end >= 0 and end >= pos[-1]):
        raise TypeError('Incorrect end format: it has to be an integer value >= pos[-1] (maximum position value)')
    return True

def check_reads_smaller_than_genomic_region(read_length, start, end):
    if read_length > end-start:
        print(end-start, read_length)
        raise TypeError('Incorrect read_length, start and end format: the genomic region defined by the start and end parameters must be larger than read_length')
    return True

def check_arc(arc):
    if not (isinstance(arc, np.ndarray) and len(arc.shape) == 3 and arc.shape[2] == 4):
        raise TypeError('Incorrect arc format: it has to be a numpy array with dimentions (sites, individuals, alleles) and the third dimention must be of size = 4')
    return True

def check_GL(GL, ploidy):
    if not (isinstance(GL, np.ndarray) and len(GL.shape) == 3):
        raise TypeError('Incorrect GL format: it has to be a numpy array with dimentions (sites, individuals, genotypes)')
    if not (len([x for x in combinations_with_replacement([0, 1, 2, 3], ploidy)]) == GL.shape[2]):
        raise TypeError('Incorrect ploidy format or GL format: the third dimention of GL {} does not correspond with the possible genotypes {} from a `ploidy` value {}'.format(GL.shape[2], array_combinations_with_replacement([0, 1, 2, 3], ploidy).size, ploidy))
    return True

def check_alleles_per_site(GL, alleles_per_site):
    if not (isinstance(alleles_per_site, np.ndarray) and len(GL.shape) == 2):
        raise TypeError('Incorrect alleles_per_site format: it has to be a numpy array with dimentions (sites, alleles_of_interest)')
    if not alleles_per_site.shape[0] == GL.shape[0]:
        raise TypeError('Incorrect alleles_per_site format or GL format: the first dimention of GL {} does not correspond with the first dimention in alleles_per_site {}'.format(GL.shape[0], alleles_per_site.shape[0]))
    return True
