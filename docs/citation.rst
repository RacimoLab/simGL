Citation
========

Software
--------

If you use simGL in published work, please cite the software repository:

    Coll Macià, M. & Gower, G. (2024). *simGL: Simulate genotype likelihoods
    from haplotypic genotype matrices* (v0.2.0). GitHub.
    https://github.com/RacimoLab/simGL

A citable Zenodo DOI will be available with the next tagged release.

Methodological references
--------------------------

Depending on which functions you use, please also cite the following papers:

**Genotype likelihood model** (:func:`simGL.allelereadcounts_to_GL`)

    Nielsen, R., Korneliussen, T., Albrechtsen, A., Li, Y., & Wang, J. (2012).
    SNP calling, genotype calling, and sample allele frequency estimation from
    new-generation sequencing data.
    *PLOS ONE*, 7(7), e37558.
    https://doi.org/10.1371/journal.pone.0037558

**Major/minor allele identification** (:func:`simGL.GL_to_Mm`)

    Skotte, L., Korneliussen, T. S., & Albrechtsen, A. (2012).
    Association testing for next-generation sequencing data using score statistics.
    *Genetic Epidemiology*, 36(5), 430–437.
    https://doi.org/10.1002/gepi.21698

**ANGSD** (the GL convention used throughout simGL)

    Korneliussen, T. S., Albrechtsen, A., & Nielsen, R. (2014).
    ANGSD: Analysis of next generation sequencing data.
    *BMC Bioinformatics*, 15, 356.
    https://doi.org/10.1186/s12859-014-0356-4
