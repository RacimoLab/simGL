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
        The values must be sorted and the order of these values must be the same as the first dimension of `gm`.
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
    
    Examples
    --------
    >>> import numpy as np
    >>> import simGL
    >>> # 3 polymorphic sites at positions 2, 5, 8 — 2 haplotypes
    >>> gm  = np.array([[0, 1],
    ...                 [1, 1],
    ...                 [0, 0]])
    >>> pos = np.array([2, 5, 8])
    >>> gm2 = simGL.incorporate_monomorphic(gm, pos, start=0, end=10)
    >>> gm2
    array([[0, 0],
           [0, 0],
           [0, 1],
           [0, 0],
           [0, 0],
           [1, 1],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0]])
    >>> gm2.shape
    (10, 2)
    '''
    _check_gm(gm)
    _check_pos(gm, pos)
    _check_start(pos, start)
    _check_end(pos, end)
    gm2 = np.zeros((int(end)-int(start), gm.shape[1]), dtype=int)
    gm2[pos.astype(int) - int(start)] = gm
    return gm2

def depth_per_haplotype(rng, mean_depth, std_depth, n_hap):
    '''
    Samples mean sequencing depth per haplotype from a gamma distribution.

    The gamma distribution captures inter-individual variation in mean coverage: in real
    sequencing data, different libraries have different depths. The sampled values are then
    used as Poisson lambda parameters in `independent_depth` to draw per-site read counts.

    If `mean_depth` is already a `numpy.ndarray`, it is returned as-is (the caller is
    providing explicit per-haplotype depths, bypassing gamma sampling).

    Parameters
    ----------
    rng : `numpy.random._generator.Generator`
        NumPy random number generator.
    mean_depth : `int`, `float`, or `numpy.ndarray`
        Mean depth per haplotype. If an array is provided it is returned unchanged.
    std_depth : `int` or `float`
        Standard deviation of the gamma distribution. Used only when `mean_depth` is scalar.
    n_hap : `int`
        Number of haplotypes to sample.

    Returns
    -------
    DPh : `numpy.ndarray`
        Mean depth per haplotype, shape (n_hap,).
    '''
    if isinstance(mean_depth, np.ndarray):
        return mean_depth
    else:
        shape = np.power(mean_depth, 2)/np.power(std_depth, 2)
        scale = np.power(std_depth, 2)/mean_depth
        return rng.gamma(shape=shape, scale=scale, size=n_hap)

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

    In this function the simulated region is treated as circular: reads extending past the end
    wrap around to the beginning, which ensures the mean coverage matches `DPh` exactly. See the
    Notes section of `sim_allelereadcounts` for a detailed explanation.
    
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
    Samples per-site read depth per haplotype from a Poisson distribution.

    For each
    haplotype, read counts at every site are independently drawn from a Poisson
    distribution whose lambda is the per-haplotype mean depth sampled by
    `depth_per_haplotype` or user provided.

    Parameters
    ----------
    rng : `numpy.random._generator.Generator`
        NumPy random number generator.
    DPh : `numpy.ndarray`
        Mean depth per haplotype, shape (haplotypes,), as returned by `depth_per_haplotype`.
    size : `tuple`
        Shape of the output array (sites, haplotypes).

    Returns
    -------
    DP : `numpy.ndarray`, dtype `int`
        Read depth per site per haplotype, shape (sites, haplotypes).
    '''
    return rng.poisson(DPh, size=size)

def sim_allelereadcounts(gm, mean_depth, e, ploidy, seed = None, std_depth = None, ref = None, alt = None, depth_type = "independent", read_length = None, start = None, end = None, pos = None):
    '''
    Simulates allele read counts from a genotype matrix. 
    
    Parameters
    ----------
    gm : `numpy.ndarray`, dtype `int`
        Genotype matrix with size (sites, haplotypic samples) in which 0 denotes reference allele
        and 1 denotes alternative allele. Only biallelic sites are supported — all values must be
        0 or 1. Multiallelic sites must be filtered out before calling this function.

    mean_depth : `int` or `float` or `numpy.ndarray`, dtype `int` or `float`
        Read depth of the each haplotypic sample in `gm`. If a `int` or `float` value is inputted, the function
        will sample random values from a gamma distribution with mean = `mean_depth` and std = `std_depth`. 

        When `mean_depth` is large relative to `std_depth`, the gamma distribution closely approximates a normal distribution with the same mean and standard deviation.
        If a `numpy.ndarray` is inputted, the array must have size (haplotypic samples, ) and the order must
        be the same as the second dimension of `gm`.
    
    std_depth : `int` or `float`
        The standard deviation parameter of the values generated by the gamma distribution from which read depth 
        values are randomly sampled for each haplotypic sample in `gm`. This value only needs to be provided if 
        the `mean_depth` inputted is an `int` or a `float`.
    
    e : `int` or `float` 
        Sequencing error probability per base pair per site. The value must be between 0 and 1.
    
    ploidy : `int` 
        Number of haplotypic chromosomes per individual.
    
    ref : `numpy.ndarray`, dtype `str`, optional
        Reference alleles list per site. The size of the array must be (sites, ) and the order has to 
        coincide with the first dimension of `gm`. The values within the list must be strings {"A", "C",
        "G", "T"}. If an `alt` list is inputted, a `ref` list must also be inputted. If no `ref` and `alt`
        are inputted, the `ref` allele is assumed to be "A" for all sites.

    alt : `numpy.ndarray`, dtype `str`, optional
        Alternative alleles list per site. The size of the array must be (sites, ) and the order has to
        coincide with the first dimension of `gm`. The values within the list must be strings {"A", "C",
        "G", "T"}. If a `ref` list is inputted, an `alt` list must also be inputted. If no `ref` and `alt`
        are inputted, the `alt` allele is assumed to be "C" for all sites.

    depth_type : `str` {`independent`, `linked`}, default = independent
        Method to simulate coverage per loci.

        - independent
            The coverage of every loci in `gm` will be independently simulated from a Poisson distribution 
            with lambda value equal to the mean depth for each haplotype.

        - linked
            Coverage is obtained by simulating a number of reads that satisfies the inputted average coverage,
            the read length and the length of the sequence, placing them randomly along the genomic region
            simulated. If this option is enabled, the arguments read_length, start, end and pos must
            also be provided. Read more about this simulation mode in the notes section below.

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
        The values must be sorted and the order of these values must be the same as the first dimension of `gm`.

    seed : `int`, optional
        Starting point in generating random numbers.
    
    Returns 
    -------
    arc : `numpy.ndarray`
        Allele read counts per site per individual. The dimensions of the array are (sites, individuals, alleles). 
        The third dimension of the array has size = 4, which corresponds to the four possible alleles: 0 = "A", 
        1 = "C", 2 = "G" and 3 = "T".
    
    Notes
    -----
    - The read depth indicated in `mean_depth` is per haplotypic sample, i.e. if the user intends to simulate a 
      depth of 30 reads per site per individual, and individuals are diploid (`ploidy` = 2), the `mean_depth` 
      must be 15. 
    - If monomorphic sites are included, the `alt` values corresponding to those sites are not taken into account,
      but they must be in the array.
    - In linked simulation mode the region is treated as circular: reads extending past the end wrap
      around to the beginning. For example, in a 10,000 bp region (positions 0–9,999), a 100 bp read
      starting at position 9,950 maps its first 50 bp to positions 9,950–9,999 and its remaining 50 bp
      to positions 0–49. This ensures the simulated mean coverage matches `mean_depth` exactly, but
      introduces a correlation between coverage at the start and end of the region.

    Examples
    --------
    >>> import numpy as np
    >>> import simGL
    >>> # 3 sites, 2 haplotypes (1 diploid individual): 0 = ref, 1 = alt
    >>> gm  = np.array([[0, 1],
    ...                 [1, 1],
    ...                 [0, 0]])
    >>> ref = np.array(["A", "C", "G"])
    >>> alt = np.array(["C", "G", "A"])
    >>> arc = simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=2.,
    ...                                  e=0.01, ploidy=2, seed=42)
    >>> arc
    array([[[ 5, 11,  0,  0]],
           [[ 0, 22,  0,  0]],
           [[27,  0,  0,  1]]])
    >>> arc.shape
    (3, 1, 4)
    '''
    #Checks
    _check_gm(gm)
    if ref is None and alt is None:
        ref = np.full(gm.shape[0], "A")
        alt = np.full(gm.shape[0], "C")
    _check_mean_depth(gm, mean_depth)
    _check_std_depth(mean_depth, std_depth)
    _check_e(e)
    _check_ploidy(ploidy)
    _check_gm_ploidy(gm, ploidy)
    _check_ref_alt(gm, ref, alt)
    _check_depth_type(depth_type)
    #Variables
    err = np.array([[1-e, e/3, e/3, e/3], [e/3, 1-e, e/3, e/3], [e/3, e/3, 1-e, e/3], [e/3, e/3, e/3, 1-e]])
    rng = np.random.default_rng(seed)
    #1. Depths (DP) per haplotype (h)
    DPh = depth_per_haplotype(rng, mean_depth, std_depth, gm.shape[1])
    #2. Sample depths (DP) per site per haplotype
    if depth_type == "independent":
        DP  = independent_depth(rng, DPh, gm.shape)
    elif depth_type == "linked":
        _check_positive_nonzero_integer(read_length, "read_length")
        _check_pos(gm, pos)
        _check_start(pos, start)
        _check_end(pos, end)
        _check_reads_smaller_than_genomic_region(read_length, start, end)
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
    '''
    Normalizes genotype likelihoods by subtracting the minimum value per site per individual.

    Parameters
    ----------
    GL : `numpy.ndarray`
        Genotype likelihoods per site per individual, shape (sites, individuals, genotypes).

    Returns
    -------
    `numpy.ndarray`
        Normalized genotype likelihoods of the same shape, where the most likely genotype
        per site per individual has value 0.

    Examples
    --------
    >>> import numpy as np
    >>> import simGL
    >>> # 1 site, 2 individuals, 3 genotypes
    >>> GL = np.array([[[3.2, 0.5, 1.1],
    ...                 [0.0, 2.3, 4.1]]])
    >>> simGL.normalize_GL(GL)
    array([[[2.7, 0. , 0.6],
            [0. , 2.3, 4.1]]])
    '''
    return GL-GL.min(axis = 2).reshape(GL.shape[0], GL.shape[1], 1)

def allelereadcounts_to_GL(arc, e, ploidy, normalized = True):
    '''
    Computes genotype likelihoods from allele read counts per site per individual. 
    
    Parameters
    ----------
    arc : `numpy.ndarray`
        Allele read counts per site per individual. The dimensions of the array are (sites, individuals, alleles). 
        The third dimension of the array has size = 4, which corresponds to the four possible alleles: 0 = "A", 
        1 = "C", 2 = "G" and 3 = "T".
    
    e : `float`
        Sequencing error probability per base pair per site. Must be strictly greater than
        0 and at most 1. A value of 0 is undefined in the GL formula because it leads to
        log(0). Error rates are typically derived from Phred-scaled base quality scores:
        for example, Q20 → e=0.01, Q30 → e=0.001 (the ANGSD default). Values below e=1e-6 (Q60) are
        rarely meaningful in practice, as differences in GLs become negligible at that scale. Note that
        this parameter can be set independently from the error rate used in `sim_allelereadcounts` —
        for example, reads can be simulated without errors (e=0) while the GL model still uses a small
        positive value (e=1e-6).

    ploidy : `int`
        Number of haplotypic chromosomes per individual.

    normalized : `bool`, default `True`
        If `True`, the returned GL array is normalized so that the most likely genotype
        per site per individual has value 0 (equivalent to calling `normalize_GL` on the
        raw output). If `False`, raw negative log-likelihoods are returned.

    Returns
    -------

    GL : `numpy.ndarray`
        Genotype likelihoods per site per individual, shape (sites, individuals, genotypes).
        The third dimension corresponds to all combinations with replacement of the 4 possible
        alleles {"A", "C", "G", "T"} (e.g. for diploids: "AA", "AC", "AG", "AT", "CC",
        "CG", ..., "TT" — 10 genotypes). Values are normalized (minimum = 0 per individual
        per site) when `normalized=True`, otherwise raw negative log-likelihoods. The GL model
        implemented here follows the fixed-error-rate simplification of the GATK genotype
        likelihood framework (1) used for population genetic inference (3), as also implemented
        in ANGSD (2).

    References
    ----------
    1) McKenna A, Hanna M, Banks E, Sivachenko A, Cibulskis K, Kernytsky A, Garimella K, Altshuler D, Gabriel S, Daly M, DePristo MA (2010). The Genome Analysis Toolkit: a MapReduce framework for analyzing next-generation DNA sequencing data. Genome Res. 20:1297-303.
    2) Thorfinn Sand Korneliussen, Anders Albrechtsen, Rasmus Nielsen. ANGSD: Analysis of Next Generation Sequencing Data. BMC Bioinform. 2014 Nov;15,356.
    3) Nielsen R, Korneliussen T, Albrechtsen A, Li Y, Wang J (2012). SNP Calling, Genotype Calling, and Sample Allele Frequency Estimation From New-Generation Sequencing Data. PLoS ONE 7(7): e37558.

    Examples
    --------
    >>> import numpy as np
    >>> import simGL
    >>> gm  = np.array([[0, 1], [1, 1], [0, 0]])
    >>> ref = np.array(["A", "C", "G"])
    >>> alt = np.array(["C", "G", "A"])
    >>> arc = simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=2., e=0.01, ploidy=2, seed=42)
    >>> GL  = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2)
    >>> GL.shape       # (sites, individuals, genotypes): 10 diploid ACGT genotypes
    (3, 1, 10)
    >>> np.round(GL[0], 3)  # site 0 (het A/C): GL=0 at index 1 (AC genotype)
    array([[51.594,  0.   , 55.043, 55.043, 17.432, 25.02 , 25.02 , 80.063,
            80.063, 80.063]])
    '''
    _check_arc(arc)
    _check_e_GL(e)
    _check_ploidy(ploidy)
    
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
        Normalized genotype likelihoods per site per individual. The dimensions of the array is (sites, individuals, genotypes). 
        The third dimension of the array corresponds to the combinations with replacement of all 4 possible alleles 
        {"A", "C", "G", "T"} (i.e., for a diploid, there are 10 possible genotypes and the combination order is "AA", "AC",
        "AG", "AT", "CC", "CG", ..., "TT"). 

    ploidy : `int` 
        Number of haplotypic chromosomes per individual. 

    Returns 
    -------
    `numpy.ndarray`
        Maximum and minimum alleles per site. The dimensions of the array is (sites, ploidy) and the values per site is an integer 
        encoding the Maximum and minimum alleles: 0 = "A", 1 = "C", 2 = "G", 3 = "T". The index of the alleles are sorted by
        order, not corresponding to the maximum and minimum alleles.
    
    Notes
    -----
    - This function requires **polymorphic sites** — i.e., sites where at least two alleles
      are observed in the sample. Fixed sites (where all individuals carry the same allele)
      provide no information about the second allele, and the returned minor allele will be
      arbitrary. Filter fixed sites before calling this function.
    - The method scores each candidate major/minor allele pair by multiplying the weighted
      GL values across all individuals (`.prod(axis=1)`). This product can underflow to zero
      when the number of individuals is large, making all pairs indistinguishable. To mitigate
      this, GL values are shifted by +1 before the product (see `get_Mmindex`). For very large
      sample sizes, consider working in log space or splitting individuals into batches.

    References
    ----------
    1) Line Skotte, Thorfinn Sand Korneliussen, Anders Albrechtsen. Association testing for next-generation sequencing data using score statistics. Genet Epidemiol. 2012 Jul;36(5):430-7.
    2) Thorfinn Sand Korneliussen, Anders Albrechtsen, Rasmus Nielsen. ANGSD: Analysis of Next Generation Sequencing Data. BMC Bioinform. 2014 Nov;15,356.

    Examples
    --------
    >>> import numpy as np
    >>> import simGL
    >>> # 2 polymorphic sites, 4 haplotypes (2 diploid individuals)
    >>> gm  = np.array([[0, 1, 0, 0],
    ...                 [0, 1, 1, 1]])
    >>> ref = np.array(["A", "G"])
    >>> alt = np.array(["C", "T"])
    >>> arc = simGL.sim_allelereadcounts(gm, mean_depth=15., std_depth=2., e=0.01, ploidy=2, seed=1, ref=ref, alt=alt)
    >>> GL  = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2)
    >>> simGL.GL_to_Mm(GL, ploidy=2)  # 0=A, 1=C, 2=G, 3=T
    array([[0, 1],
           [2, 3]])
    '''
    _check_ploidy(ploidy)
    _check_GL(GL, ploidy)
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
        Normalized genotype likelihoods per site per individual. The dimensions of the array is (sites, individuals, genotypes). 
        The third dimension of the array corresponds to the combinations with replacement of all 4 possible alleles 
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
        Subset of the inputted `GL` matrix with the likelihoods corresponding to the queried alleles.

    Examples
    --------
    >>> import numpy as np
    >>> import simGL
    >>> # 2 polymorphic sites, 4 haplotypes (2 diploid individuals)
    >>> gm  = np.array([[0, 1, 0, 0],
    ...                 [0, 1, 1, 1]])
    >>> ref = np.array(["A", "G"])
    >>> alt = np.array(["C", "T"])
    >>> arc = simGL.sim_allelereadcounts(gm, mean_depth=15., std_depth=2., e=0.01, ploidy=2, seed=1, ref=ref, alt=alt)
    >>> GL  = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2)
    >>> GL.shape  # (sites, individuals, genotypes): 10 diploid ACGT genotypes
    (2, 2, 10)
    >>> Mm = simGL.GL_to_Mm(GL, ploidy=2)  # 0=A, 1=C, 2=G, 3=T
    >>> Mm
    array([[0, 1],
           [2, 3]])
    >>> np.round(simGL.subset_GL(GL, Mm, ploidy=2), 3)
    array([[[ 84.038,   0.   ,  66.957],
            [  0.   ,  17.934, 148.037]],
           [[ 53.149,   0.   ,  41.762],
            [244.83 ,  29.661,   0.   ]]])
    '''
    _check_ploidy(ploidy)
    _check_GL(GL, ploidy)
    _check_alleles_per_site(GL, alleles_per_site)
    GTxploidy = array_combinations_with_replacement([0, 1, 2, 3], ploidy)

    n_loci, n_ind, n_GL = GL.shape
    n_genotypes = array_combinations_with_replacement(np.arange(alleles_per_site.shape[1]), ploidy).shape[0]

    dim1 = np.repeat(np.arange(n_loci), n_genotypes*n_ind)
    dim2 = np.repeat(np.tile(np.arange(n_ind), n_loci), n_genotypes)

    GTidxxn_loci = np.tile(np.arange(n_GL), n_loci).reshape(n_loci, n_GL) #Construct a dummy matrix with dimensions (n_loci, n_GL) that will have the index of each GT
    GTidxxn_loci_bool = ((alleles_per_site.reshape(n_loci, ploidy, 1, 1) == GTxploidy.T.reshape(1, 1, ploidy, n_GL)).sum(axis = 2).sum(axis = 1) == ploidy) # boolean matrix with dimensions (n_loci, n_GL) that encodes which GT positions correspond to the Mm alleles
    dim3 = np.tile(GTidxxn_loci[GTidxxn_loci_bool].reshape(n_loci, -1), n_ind).reshape(-1) #Array with the GT index positions to retrieve, repeated (tiled) as many times as n_ind

    GL_subset = GL[dim1, dim2, dim3].reshape(n_loci, n_ind, n_genotypes)

    # Genotypes are retrieved in sorted allele-index order internally. When the caller
    # passed alleles in descending order (e.g. [G=2, C=1]), reorder the output so that
    # index 0 = hom for first allele, index 1 = het, index 2 = hom for second allele.
    correspondance, unsorted_rows = sorted_unsorted_allele_combination_correspondance(alleles_per_site, ploidy)
    n_unsorted_rows = unsorted_rows.sum()

    if n_unsorted_rows > 0:
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
        Allele read counts per site per individual. The dimensions of the array are (sites, individuals, alleles). 
        The third dimension of the array has size = 4, which corresponds to the four possible alleles: 0 = "A", 
        1 = "C", 2 = "G" and 3 = "T".
    
    output : `str`
        Output file name.

    Returns 
    -------
    None
    '''
    if not (isinstance(arc, np.ndarray) and len(arc.shape) == 3 and arc.shape[2] == 4):
        raise TypeError('Incorrect `arc` format: it has to be a numpy array with dimensions (sites, individuals, alleles) and the third dimension must be of size = 4')
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

def ref_alt_to_index(ref, alt):
    '''
    Converts reference and alternative allele strings to integer indices.

    A convenience function that produces an array suitable for passing directly
    to `subset_GL`. Encoding: 0 = "A", 1 = "C", 2 = "G", 3 = "T".

    Parameters
    ----------
    ref : `numpy.ndarray`, dtype `str`
        Reference alleles per site, shape (sites,). Values must be in {"A", "C", "G", "T"}.
    alt : `numpy.ndarray`, dtype `str`
        Alternative alleles per site, shape (sites,). Values must be in {"A", "C", "G", "T"}.

    Returns
    -------
    `numpy.ndarray`, dtype `int`
        Integer-encoded ref/alt allele pairs, shape (sites, 2). First column is the
        reference allele index, second is the alternative allele index.

    Examples
    --------
    >>> ref = np.array(["A", "C", "G"])
    >>> alt = np.array(["T", "G", "A"])
    >>> simGL.ref_alt_to_index(ref, alt)
    array([[0, 3],
           [1, 2],
           [2, 0]])
    '''
    if not (isinstance(ref, np.ndarray) and isinstance(alt, np.ndarray)):
        raise TypeError("ref and alt must be numpy arrays")
    if ref.ndim != 1 or ref.shape != alt.shape:
        raise TypeError("ref and alt must be 1-D arrays of the same length")
    valid = {"A", "C", "G", "T"}
    if not (set(np.unique(ref)) <= valid and set(np.unique(alt)) <= valid):
        raise TypeError('ref and alt values must be in {"A", "C", "G", "T"}')
    base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    ref_idx = np.array([base_to_int[b] for b in ref])
    alt_idx = np.array([base_to_int[b] for b in alt])
    return np.stack([ref_idx, alt_idx], axis=1)

def GL_to_vcf(GL_subset, arc, ref, alt, pos, sample_names, output, chrom="1", contig_length=None):
    '''
    Writes genotype likelihoods and allele read counts to a VCF 4.2 file.

    The FORMAT field written is ``GT:GL:AD``. Genotype calls are derived from the
    normalized GL values: the genotype with value 0 is called; if more than one
    value is 0 (zero coverage), the genotype is set to missing (``./.``).

    Parameters
    ----------
    GL_subset : `numpy.ndarray`
        Normalized biallelic genotype likelihoods, shape (sites, individuals, 3)
        for diploids — as returned by `subset_GL` followed by `normalize_GL`.
    arc : `numpy.ndarray`
        Allele read counts, shape (sites, individuals, 4). Third dimension order:
        0 = "A", 1 = "C", 2 = "G", 3 = "T".
    ref : `numpy.ndarray`, dtype `str`
        Reference alleles per site, shape (sites,).
    alt : `numpy.ndarray`, dtype `str`
        Alternative alleles per site, shape (sites,).
    pos : `numpy.ndarray`, dtype `int`
        1-based genomic positions per site, shape (sites,).
    sample_names : array-like of `str`
        Sample identifiers. Length must equal the number of individuals in `GL_subset`.
    output : `str`
        Output VCF file path.
    chrom : `str`, default ``"1"``
        Chromosome/contig name written to the CHROM column.
    contig_length : `int` or `None`, default `None`
        If provided, a ``##contig`` header line is written with this length.

    Returns
    -------
    None

    Notes
    -----
    - GL values are rounded to 3 decimal places in the output.
    - GT is called from the index of the 0-valued GL entry:
      index 0 → ``0/0``, index 1 → ``0/1``, index 2 → ``1/1``.
      If more than one GL value equals 0 (e.g. zero coverage), the genotype
      is written as ``./.``.
    - Currently supports diploid biallelic output only (3 genotypes per individual).

    Examples
    --------
    >>> import numpy as np
    >>> import simGL
    >>> gm  = np.array([[0, 1], 
                        [1, 1], 
                        [0, 0], 
                        [0, 1]])
    >>> ref = np.array(["A", "C", "G", "T"])
    >>> alt = np.array(["C", "G", "A", "A"])
    >>> pos = np.array([100, 200, 300, 400])
    >>> arc = simGL.sim_allelereadcounts(gm, mean_depth=10., std_depth=2., e=0.01, ploidy=2, seed=42)
    >>> GL  = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2, normalized=False)
    >>> Ra  = simGL.ref_alt_to_index(ref, alt)
    >>> GL_sub = simGL.normalize_GL(simGL.subset_GL(GL, Ra, ploidy=2))
    >>> simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos, ["ind0"], "out.vcf")
    >>> print(open("out.vcf").read())
    ##fileformat=VCFv4.2
    ##FILTER=<ID=PASS,Description="All filters passed">
    ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
    ##FORMAT=<ID=GL,Number=G,Type=Float,Description="Normalized genotype likelihoods (negative log scale)">
    ##FORMAT=<ID=AD,Number=4,Type=Integer,Description="Read depth per allele: 0=A, 1=C, 2=G, 3=T">
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	ind0
    1	100	.	A	C	.	.	.	GT:GL:AD	0/1:45.901,0.0,23.126:6,10,0,0
    1	200	.	C	G	.	.	.	GT:GL:AD	0/0:0.0,14.486,119.568:1,21,0,0
    1	300	.	G	A	.	.	.	GT:GL:AD	1/1:159.424,19.314,0.0:28,0,0,0
    1	400	.	T	A	.	.	.	GT:GL:AD	1/1:74.019,8.967,0.0:13,11,0,0
    '''
    if not (isinstance(GL_subset, np.ndarray) and GL_subset.ndim == 3):
        raise TypeError("GL_subset must be a 3-D numpy array (sites, individuals, genotypes)")
    if not (isinstance(arc, np.ndarray) and arc.ndim == 3 and arc.shape[2] == 4):
        raise TypeError("arc must be a numpy array with shape (sites, individuals, 4)")
    if GL_subset.shape[0] != arc.shape[0] or GL_subset.shape[1] != arc.shape[1]:
        raise TypeError("GL_subset and arc must have the same number of sites and individuals")
    if not (isinstance(ref, np.ndarray) and isinstance(alt, np.ndarray) and ref.ndim == 1 and ref.shape == alt.shape and ref.shape[0] == GL_subset.shape[0]):
        raise TypeError("ref and alt must be 1-D numpy arrays with length equal to the number of sites")
    if not (isinstance(pos, np.ndarray) and pos.ndim == 1 and pos.shape[0] == GL_subset.shape[0]):
        raise TypeError("pos must be a 1-D numpy array with length equal to the number of sites")
    sample_names = list(sample_names)
    if len(sample_names) != GL_subset.shape[1]:
        raise TypeError("sample_names length must equal the number of individuals in GL_subset")
    if not isinstance(output, str):
        raise TypeError("output must be a string file path")

    _gt_map = {0: "0/0", 1: "0/1", 2: "1/1"}

    header_lines = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=GL,Number=G,Type=Float,Description="Normalized genotype likelihoods (negative log scale)">',
        '##FORMAT=<ID=AD,Number=4,Type=Integer,Description="Read depth per allele: 0=A, 1=C, 2=G, 3=T">',
    ]
    if contig_length is not None:
        header_lines.append(f'##contig=<ID={chrom},length={contig_length}>')

    col_header = "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + sample_names)

    n_sites, n_ind = GL_subset.shape[0], GL_subset.shape[1]

    with open(output, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        f.write(col_header + "\n")
        for i in range(n_sites):
            fixed = f"{chrom}\t{pos[i]}\t.\t{ref[i]}\t{alt[i]}\t.\t.\t.\tGT:GL:AD"
            sample_fields = []
            for j in range(n_ind):
                gl_vals = GL_subset[i, j]
                zero_mask = gl_vals == 0.0
                if zero_mask.sum() != 1:
                    gt = "./."
                else:
                    gt = _gt_map.get(int(np.where(zero_mask)[0][0]), "./.")
                gl_str = ",".join(str(round(float(x), 3)) for x in gl_vals)
                ad_str = ",".join(str(int(x)) for x in arc[i, j])
                sample_fields.append(f"{gt}:{gl_str}:{ad_str}")
            f.write(fixed + "\t" + "\t".join(sample_fields) + "\n")


# Functions to check input formatting
def _check_gm(gm):
    if not (isinstance(gm, np.ndarray) and len(gm.shape) == 2 and ((gm == 0)+(gm == 1)).sum() == gm.size):
        raise TypeError('Incorrect gm format: it has to be a numpy array with dimensions (sites, haplotypic samples) with integer values 1 and 0')

def _check_mean_depth(gm, mean_depth):
    if not ((isinstance(mean_depth, np.ndarray) and len(mean_depth.shape) == 1 and mean_depth.shape[0] == gm.shape[1] and (mean_depth > 0).sum() == mean_depth.size) or (isinstance(mean_depth, (int, float)) and mean_depth > 0.0)):
        raise TypeError('Incorrect mean_depth format: it has to be either i) numpy.array with dimensions (haplotypic samples, ) with values > 0 or ii) integer or float value > 0')

def _check_std_depth(mean_depth, std_depth):
    if not ((isinstance(mean_depth, np.ndarray)) or (isinstance(std_depth, (int, float)) and std_depth >= 0.0)):
        raise TypeError('Incorrect std_depth format: it has to be an integer or float value > 0 if mean_depth is a integer or float value and not a numpy array')

def _check_e(e):
    if not (isinstance(e, (int, float)) and e >= 0.0 and e <= 1.0):
        raise TypeError('Incorrect e format: it has to be a float value >= 0 and <= 1')

def _check_e_GL(e):
    if not (isinstance(e, (int, float)) and e > 0.0 and e <= 1.0):
        raise TypeError('Incorrect e format: it has to be a float value > 0 and <= 1. '
                        'A value of 0 is undefined in the GL formula (leads to log(0)). '
                        'Use a Phred-derived value such as e=0.01 (Q20) or e=0.001 (Q30).')

def _check_ploidy(ploidy):
    if not (isinstance(ploidy, int) and ploidy > 0) :
        raise TypeError('Incorrect ploidy format: it has to be an integer value > 0')

def _check_gm_ploidy(gm, ploidy):
    if not (gm.shape[1]%ploidy == 0) :
        raise TypeError('Incorrect ploidy and/or gm format: the second dimension of gm (haplotypic samples) must be divisible by ploidy')

def _check_depth_type(depth_type):
    if not isinstance(depth_type, str) or depth_type not in ["independent", "linked"]:
        raise TypeError('Incorrect depth_type format: it has to be a string, either "independent" or "linked"')

def _check_positive_nonzero_integer(value, name):
    if not isinstance(value, int) or value <= 0:
        raise TypeError('Incorrect {} format: it has to be a integer value > 0'.format(name))

def _check_ref_alt(gm, ref, alt):
    if not (isinstance(ref, np.ndarray) and isinstance(alt, np.ndarray) and len(ref.shape) == 1 and len(alt.shape) == 1 and ref.shape == alt.shape and ref.size == gm.shape[0] and
              ((ref == "A") + (ref == "C") + (ref == "G") + (ref == "T")).sum() == ref.size and ((alt == "A") + (alt == "C") + (alt == "G") + (alt == "T")).sum() == alt.size):
        raise TypeError('Incorrect ref and/or alt format: they both have to be a numpy array with dimensions (sites, ) with string "A", "C", "G", "T" values')

def _check_pos(gm, pos):
    if not (isinstance(pos, np.ndarray) and len(pos.shape) == 1 and (pos >= 0).sum() == pos.size and pos.shape[0] == gm.shape[0] and np.issubdtype((pos).dtype, np.integer) and (pos[:-1] >= pos[1:]).sum() == 0):        
        raise TypeError('Incorrect pos format: it has to be a sorted numpy array with dimensions (polymorphic sites, ) corresponding to the genomic coordinates of loci')

def _check_start(pos, start):
    if not (isinstance(start, int) and start >= 0 and start <= pos[0]):
        raise TypeError('Incorrect start format: it has to be an integer value >=0 and <= pos[0] (minimum position value) ')

def _check_end(pos, end):
    if not (isinstance(end, int) and end >= 0 and end >= pos[-1]):
        raise TypeError('Incorrect end format: it has to be an integer value >= pos[-1] (maximum position value)')

def _check_reads_smaller_than_genomic_region(read_length, start, end):
    if read_length > end-start:
        raise TypeError('Incorrect read_length, start and end format: the genomic region defined by the start and end parameters must be larger than read_length')

def _check_arc(arc):
    if not (isinstance(arc, np.ndarray) and len(arc.shape) == 3 and arc.shape[2] == 4):
        raise TypeError('Incorrect arc format: it has to be a numpy array with dimensions (sites, individuals, alleles) and the third dimension must be of size = 4')

def _check_GL(GL, ploidy):
    if not (isinstance(GL, np.ndarray) and len(GL.shape) == 3):
        raise TypeError('Incorrect GL format: it has to be a numpy array with dimensions (sites, individuals, genotypes)')
    if not (len([x for x in combinations_with_replacement([0, 1, 2, 3], ploidy)]) == GL.shape[2]):
        raise TypeError('Incorrect ploidy format or GL format: the third dimension of GL {} does not correspond with the possible genotypes {} from a `ploidy` value {}'.format(GL.shape[2], array_combinations_with_replacement([0, 1, 2, 3], ploidy).size, ploidy))

def _check_alleles_per_site(GL, alleles_per_site):
    if not (isinstance(alleles_per_site, np.ndarray) and len(alleles_per_site.shape) == 2):
        raise TypeError('Incorrect alleles_per_site format: it has to be a numpy array with dimensions (sites, alleles_of_interest)')
    if not alleles_per_site.shape[0] == GL.shape[0]:
        raise TypeError('Incorrect alleles_per_site format or GL format: the first dimension of GL {} does not correspond with the first dimension in alleles_per_site {}'.format(GL.shape[0], alleles_per_site.shape[0]))