Quickstart
==========

This guide walks through a minimal end-to-end example: simulate a tree sequence,
simulate read counts based on coverage and error rates, compute genotype likelihoods
from those, and write a VCF file.

Simulate a tree sequence
------------------------

The first step is to simulate a tree sequence. Here we use msprime to obtain a
haplotypic genotype matrix of shape ``(n_sites, n_haplotypes)`` that serves as
input to simGL.

simGL requires the input matrix to be biallelic (values 0 or 1 only), so we
filter out any site where more than two alleles are present:

.. code-block:: python

   import msprime
   import numpy as np

   ts = msprime.sim_ancestry(
       samples=10, ploidy=2,
       sequence_length=100_000,
       recombination_rate=1e-8,
       population_size=10_000,
       random_seed=1,
   )
   ts = msprime.sim_mutations(ts, rate=1e-4, random_seed=1)

   # Keep only biallelic sites (required by simGL)
   gm_full = ts.genotype_matrix()           # shape (n_sites, n_haplotypes)
   biallelic = gm_full.max(axis=1) == 1
   gm = gm_full[biallelic]                  # shape (n_biallelic_sites, n_haplotypes)

Each row of ``gm`` is a genomic site; each column is a haplotype (two per diploid
individual). Entries are 0 (reference allele) and 1 (alternative allele). For
example, with 3 individuals (6 haplotypes):

.. code-block:: text

   gm (first 3 sites):
   [[1 1 1 0 0 0],   # site 0: ind0=hom-alt, ind1=het,     ind2=hom-ref
    [1 0 0 1 1 1],   # site 1: ind0=het,     ind1=het,     ind2=hom-alt
    [1 0 1 0 0 0]]   # site 2: ind0=het,     ind1=het,     ind2=hom-ref

Get reference and alternative alleles
--------------------------------------

Extract the reference and alternative allele labels from the tree sequence before
simulating reads. These are needed both for read simulation and for labelling the
output VCF:

.. code-block:: python

   variants = list(ts.variants())
   ref = np.array([v.alleles[0] for v in variants])[biallelic]
   alt = np.array([v.alleles[1] for v in variants])[biallelic]

Simulate read counts
--------------------

Once we have the genotype matrix and allele labels, simulate read counts for each
individual at each site. The output ``arc`` has shape
``(n_sites, n_individuals, 4)``, where the last dimension holds read counts for
A, C, G, T in that order:

.. code-block:: python

   import simGL

   arc = simGL.sim_allelereadcounts(
       gm,
       mean_depth=10.0,
       std_depth=2.0,
       e=0.01,       # sequencing error rate for read simulation
       ploidy=2,
       seed=42,
       ref=ref,
       alt=alt,
   )
   # arc shape: (n_sites, n_individuals, 4)  — counts for A, C, G, T

For example, ``arc[0]`` for site 0 with ref=T and alt=A:

.. code-block:: text

   arc[0]:
   [[17  0  0  0],   # ind0: 17 alt (A) reads — hom-alt
    [ 7  0  1  9],   # ind1: 7 alt (A), 9 ref (T), 1 error read — het
    [ 0  0  1 15]]   # ind2: 15 ref (T), 1 error read — hom-ref

**Specifying coverage per individual.** By default simGL draws one mean depth per
haplotype from a gamma distribution with the given ``mean_depth`` and ``std_depth``.
To reproduce a specific per-individual coverage profile, for example, if the user
is simulating a specific dataset for which the coverage per individual is known, 
pass a 1-D array of mean depths with one entry per haplotype. Note that the 
coverage is per haplotype, therefore, for diploid data, the array should have two 
entries per individual (each having half the individual's mean depth):

.. code-block:: python

   mean_depths = np.array([5., 5., 10., 10., 3., 3.,   # individuals 0–2
                            5., 5., 10., 10., 3., 3.,   # individuals 3–5
                            5., 5., 10., 10., 3., 3.,   # individuals 6–8
                            5., 5.])                     # individual 9
   arc_perind = simGL.sim_allelereadcounts(
       gm, mean_depth=mean_depths, std_depth=0.,
       e=0.01, ploidy=2, seed=1, ref=ref, alt=alt,
   )

**Simulating coverage correlations across sites (linked mode).** By default
coverage is sampled independently at each site from a Poisson distribution.
Setting ``depth_type="linked"`` instead places whole reads of a fixed length (``read_length``)
randomly across the region, so that nearby sites share coverage from overlapping reads:

.. code-block:: python

   pos = np.array([int(v.site.position) for v in variants])[biallelic]

   arc_linked = simGL.sim_allelereadcounts(
       gm, mean_depth=10., std_depth=2., e=0.01, ploidy=2, seed=1,
       ref=ref, alt=alt,
       depth_type="linked", read_length=100,
       start=0, end=100_000, pos=pos,
   )

Compute genotype likelihoods (GL)
----------------------------------

With the allele read counts, compute genotype likelihoods (GLs) for each
individual at each site. Likelihoods are computed for all possible diploid
genotypes (10 for diploids organisms with four-allele ACGT data), giving shape
``(n_sites, n_individuals, 10)``:

.. code-block:: python

   GL = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2)
   # GL shape: (n_sites, n_individuals, 10)  — one value per diploid ACGT genotype

For example, ``GL[0]`` for site 0 (ref=T, alt=A). The 10 genotype columns follow
the order AA, AC, AG, AT, CC, CG, CT, GG, GT, TT:

.. code-block:: text

   GL[0]:
   #          AA      AC      AG      AT      CC      CG      CT      GG      GT      TT
   ind0:  [  0.0 ,  11.7 ,  11.7 ,  11.7 ,  96.8 ,  96.8 ,  96.8 ,  96.8 ,  96.8 ,  96.8 ]  # hom-alt (AA)
   ind1:  [ 40.2 ,  45.0 ,  40.0 ,   0.0 ,  80.1 ,  75.1 ,  35.0 ,  74.4 ,  30.0 ,  28.8 ]  # het (AT)
   ind2:  [ 85.4 ,  85.4 ,  80.4 ,  10.3 ,  85.4 ,  80.4 ,  10.3 ,  79.7 ,   5.3 ,   0.0 ]  # hom-ref (TT)

Each row is normalized so that the minimum (most likely genotype) equals 0.
This can be prevented by setting ``normalized=False`` in the function call.

Subset to biallelic GLs and write a VCF
-----------------------------------------

The full genotype GL is rarely needed downstream. Use :func:`simGL.subset_GL` to reduce it to the
three biallelic genotypes (e.g., hom-ref, het, hom-alt):

.. code-block:: python

   Ra      = simGL.ref_alt_to_index(ref, alt)        # shape (n_sites, 2)
   GL_sub  = simGL.subset_GL(GL, Ra, ploidy=2)       # shape (n_sites, n_ind, 3)

For example, ``GL_sub[0]`` for site 0:

.. code-block:: text

   GL_sub[0]:
   #           hom-ref   het   hom-alt
   ind0:  [  96.8 ,  11.7 ,   0.0 ]   # hom-alt most likely
   ind1:  [  28.8 ,   0.0 ,  40.2 ]   # het most likely
   ind2:  [   0.0 ,  10.3 ,  85.4 ]   # hom-ref most likely

If reference and alternative alleles are not available, use
:func:`simGL.GL_to_Mm` to identify the major and minor alleles directly from the
GL array, then pass the result to :func:`simGL.subset_GL`:

.. code-block:: python

   Mm     = simGL.GL_to_Mm(GL, ploidy=2)       # shape (n_sites, 2) — allele indices
   GL_sub = simGL.subset_GL(GL, Mm, ploidy=2)
   GL_sub = simGL.normalize_GL(GL_sub)

Finally, write a VCF file with genotype calls, likelihoods, and allele depths:

.. code-block:: python

   pos   = np.array([int(v.site.position) for v in variants])[biallelic] + 1
   names = [f"ind{i}" for i in range(ts.num_individuals)]

   simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos, names, "output.vcf")

The resulting ``output.vcf`` contains ``GT:GL:AD`` fields for every sample at
every biallelic site. The first two data lines look like:

.. code-block:: text

   #CHROM  POS  ID  REF  ALT  QUAL  FILTER  INFO  FORMAT   ind0                        ind1                      ind2
   1       59   .   T    A    .     .       .     GT:GL:AD  1/1:96.8,11.7,0.0:17,0,0,0  0/1:28.8,0.0,40.2:7,0,1,9  0/0:0.0,10.3,85.4:0,0,1,15
   1       114  .   T    A    .     .       .     GT:GL:AD  0/1:62.5,0.0,45.4:14,0,0,11  0/1:39.5,0.0,33.8:9,0,0,8  1/1:113.9,13.8,0.0:20,0,0,0
