User Guide
==========

This page describes the main functions of simGL and the considerations
that govern how they are used together.

Genotype matrix requirements
-----------------------------

simGL operates on *biallelic* sites only. The ``gm`` array passed to
:func:`simGL.sim_allelereadcounts` must contain only 0 and 1 values
(0 = carries reference allele, 1 = carries alternate allele).

When using `msprime <https://tskit.dev/msprime/>`_ to generate input, the
raw genotype matrix can contain values greater than 1 at sites where multiple
independent mutations occur at the same genomic position (recurrent mutation).
Filter these out before calling simGL:

.. code-block:: python

   gm_full = ts.genotype_matrix()
   gm = gm_full[gm_full.max(axis=1) == 1]

Coverage model
--------------

There are two ways to specify coverage for the simulated haplotypes:

1. **Single value.** Provide a scalar ``mean_depth`` and ``std_depth``. simGL
   draws a mean depth for each haplotype from a gamma distribution with the
   specified mean and standard deviation.

2. **Per-haplotype vector.** Provide a 1-D array with one entry per haplotype.
   simGL uses those values directly as the per-haplotype Poisson means, which
   is useful for reproducing a specific coverage profile from real data.

Once the number of reads per haplotype per site is determined, reads are
distributed across the four possible bases (A, C, G, T) according to the
genotype of the individual at that site and the sequencing error rate ``e``.
Setting ``e = 0`` is valid here — it simply means no read errors are introduced.

Read counts can also be generated in two ways, controlled by the ``depth_type``
argument:

1. **Independent (default).** Read counts at each site are sampled independently
   from a Poisson distribution, assuming coverage is uncorrelated across sites.

2. **Linked.** simGL places whole reads of a fixed length randomly along each
   haplotype. Read counts at a site are then obtained by counting how many reads
   overlap that position. This introduces correlation between coverage at nearby
   sites and is more realistic for short-read data.

Genotype likelihood computation
---------------------------------

:func:`simGL.allelereadcounts_to_GL` computes genotype log-likelihoods using
the ANGSD/GATK formula. The sequencing error rate ``e`` appears inside a
logarithm, so **e must be strictly greater than zero**. The ANGSD default
(Phred Q30) is ``e = 0.001``. If the user wants to simulate error-free reads,
set ``e`` to a very small value (e.g. ``1e-6``) instead of zero.

The output GL array has shape ``(n_sites, n_individuals, n_genotypes)``
where ``n_genotypes = 10`` for diploid data (all ACGT genotype pairs).
Values are *negative* log-likelihoods; a smaller value means a more likely
genotype.

Subsetting to biallelic genotypes
----------------------------------

The full vector of GL is rarely needed downstream. Use
:func:`simGL.ref_alt_to_index` and :func:`simGL.subset_GL` to reduce it to
the three biallelic genotypes (hom-ref, het, hom-alt):

.. code-block:: python

   Ra     = simGL.ref_alt_to_index(ref, alt)   # shape (n_sites, 2)
   GL_sub = simGL.subset_GL(GL, Ra, ploidy=2)  # shape (n_sites, n_ind, 3)

``subset_GL`` always places the hom-ref genotype at index 0, het at 1,
and hom-alt at 2, regardless of whether the reference allele has a
higher or lower ACGT index than the alternate allele.

If reference and alternative allele information is not available, use
:func:`simGL.GL_to_Mm` to identify the major and minor alleles from the GL
array and pass the result to :func:`simGL.subset_GL`. See below for details.

Normalization
-------------

:func:`simGL.normalize_GL` shifts each individual's GL vector so that
the best (minimum) value becomes 0. This is required before writing a
VCF file and is consistent with the convention used by other software such
as GATK and ANGSD.

.. code-block:: python

   GL_norm = simGL.normalize_GL(GL)

Minor-allele frequency estimation
-----------------------------------

:func:`simGL.GL_to_Mm` identifies the most likely major and minor alleles
at each site by scoring every candidate allele pair with a likelihood-based
criterion: for each pair it computes a HWE-weighted combination of the GL
values across all individuals, then picks the pair with the best (minimum)
combined score. It requires **polymorphic sites only** (i.e. sites where the
alternate allele frequency is strictly between 0 and 1). Fixed sites
(all ref or all alt) must be removed before calling this function.

Writing a VCF file
------------------

:func:`simGL.GL_to_vcf` writes a VCF 4.2 file with the following
FORMAT fields:

* ``GT`` — genotype call derived from the argmin of the normalized GL
  (missing ``./.`` when all GL values are equal, i.e. no reads).
* ``GL`` — normalized biallelic genotype log-likelihoods (3 values per
  sample, rounded to 3 decimal places).
* ``AD`` — allele depth for all four ACGT alleles, taken directly from
  the ``arc`` read-count array.

Writing a pileup file
---------------------

:func:`simGL.allelereadcounts_to_pileup` converts an ``arc`` array to a
samtools-style pileup file. Each row corresponds to one genomic site;
each individual contributes three columns (depth, base string, quality string).
This format can be used as input to tools that expect pileup data.

Monomorphic sites
-----------------

Genotype matrices produced by simulation software typically contain only
polymorphic sites. To simulate sequencing reads for monomorphic positions as
well (e.g. to model random sequencing errors at invariant sites),
:func:`simGL.incorporate_monomorphic` inserts rows in which all haplotypes
carry the reference allele (value 0) at every position not already present in
the polymorphic matrix. The expanded matrix can then be passed directly to
:func:`simGL.sim_allelereadcounts`.

Be aware that a larger matrix increases runtime and memory use, so insert
monomorphic sites only when strictly necessary.

.. code-block:: python

   gm_full = simGL.incorporate_monomorphic(gm, pos, start=0, end=100_000)
