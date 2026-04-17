Background and Theory
=====================

This page gives a brief overview of the statistical models that underpin simGL.

Coverage simulation
-------------------

Let :math:`n` be the number of individuals and :math:`S` the number of sites.
For individual :math:`i`, the per-individual mean depth is sampled once from a
Gamma distribution:

.. math::

   \lambda_i \sim \text{Gamma}\!\left(\frac{\mu^2}{\sigma^2},\,\frac{\sigma^2}{\mu}\right)

where :math:`\mu` = ``mean_depth`` and :math:`\sigma` = ``std_depth``.
The Gamma parameterisation used here has mean :math:`\mu` and variance
:math:`\sigma^2`.

For each site :math:`s` of individual :math:`i`, the read count is then:

.. math::

   d_{s,i} \sim \text{Poisson}(\lambda_i)


Read simulation
---------------

Given a diploid genotype :math:`g \in \{0\text{/}0,\,0\text{/}1,\,1\text{/}1\}`
at a biallelic site, the true allele probability for a randomly sampled read is:

.. math::

   p(\text{ref} \mid g) = \frac{\text{number of ref haplotypes in }g}{2}

With sequencing error rate :math:`e`, each read is flipped to a uniformly
random wrong base with probability :math:`e` and retained as the true base
with probability :math:`1 - e`.

Genotype likelihoods
--------------------

Given :math:`d` observed reads, let :math:`k_a` be the count of reads for
allele :math:`a \in \{A, C, G, T\}`.  For a diploid genotype
:math:`\{a_1, a_2\}` the log-likelihood under the ANGSD model is:

.. math::

   \log P(\text{reads} \mid a_1 a_2) =
   \sum_{b \in \{A,C,G,T\}} k_b \log
   \left[
     \frac{1}{2}\, p(b \mid a_1,\, e)
     + \frac{1}{2}\, p(b \mid a_2,\, e)
   \right]

where

.. math::

   p(b \mid a,\, e) =
   \begin{cases}
     1 - e & \text{if } b = a \\
     e / 3  & \text{if } b \neq a
   \end{cases}

Because :math:`e` appears inside a logarithm, it must be strictly positive.
Setting :math:`e = 0` would give :math:`\log 0 = -\infty` for any read that
does not match the assumed allele.

The ANGSD default corresponds to Phred quality Q30, i.e.
:math:`e = 10^{-30/10} \approx 0.001`.


Minor allele frequency estimation
----------------------------------

:func:`simGL.GL_to_Mm` identifies the most likely major (M) and minor (m)
allele pair at each site.  For every candidate pair :math:`(M, m)`, the
function scores individuals by combining their per-individual GL values
with the prior probability of each genotype under Hardy–Weinberg equilibrium
(equal allele frequencies, :math:`p = 0.5`), then multiplies those scores
across individuals.  The pair with the minimum combined score (highest
likelihood) is returned.  This approach follows the scoring framework of
`Skotte et al. (2012) <https://doi.org/10.1002/gepi.21698>`_ and
`Nielsen et al. (2012) <https://doi.org/10.1371/journal.pgen.1003003>`_.
