Changelog
=========

0.2.0 (unreleased)
-------------------

New functions
~~~~~~~~~~~~~

* :func:`simGL.ref_alt_to_index` — converts reference and alternate allele
  strings to ACGT integer indices, for use with :func:`simGL.subset_GL`.
* :func:`simGL.GL_to_vcf` — writes a VCF 4.2 file from normalized biallelic
  GLs and allele read counts, with ``GT``, ``GL``, and ``AD`` FORMAT fields.

Bug fixes
~~~~~~~~~

* :func:`simGL.incorporate_monomorphic`: fixed incorrect indexing when
  ``start > 0``.  The output array is sized ``end - start`` but absolute
  genomic positions were previously used as array indices.
* :func:`simGL.subset_GL`: fixed an ``IndexError`` when all allele pairs
  were already in ascending order (the empty index array defaulted to
  ``float64`` and could not be used as an integer index).

Improvements
~~~~~~~~~~~~

* :func:`simGL.allelereadcounts_to_GL`: the error rate ``e`` must now be
  strictly greater than zero.  A new validator raises ``TypeError`` for
  ``e = 0`` with an informative message referencing the ANGSD default.
* Extended docstrings for all public functions, including parameter
  descriptions, return types, and usage notes.
* Fixed spelling errors (``dimetion`` → ``dimension``,
  ``inputed`` → ``inputted``, ``beggining`` → ``beginning``) throughout
  the source code.

0.1.0
-----

Initial release.
