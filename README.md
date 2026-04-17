# simGL

**simGL** simulates genotype likelihoods (GLs) from haplotypic genotype matrices,
given per-sample coverage and sequencing error rates.
It is designed to work seamlessly with
[msprime](https://tskit.dev/msprime/) and [tskit](https://tskit.dev/tskit/)
pipelines, but accepts any NumPy haplotype matrix.

## Installation

```bash
pip install simGL
```

Or from source:

```bash
git clone https://github.com/RacimoLab/simGL.git
cd simGL
pip install -e .
```

## Quick example

```python
import msprime
import numpy as np
import simGL

# 1. Simulate a tree sequence and extract the biallelic genotype matrix
ts = msprime.sim_ancestry(
    samples=10, ploidy=2, sequence_length=100_000,
    recombination_rate=1e-8, population_size=10_000, random_seed=1,
)
ts = msprime.sim_mutations(ts, rate=1e-4, random_seed=1)

gm_full  = ts.genotype_matrix()
biallelic = gm_full.max(axis=1) == 1
gm       = gm_full[biallelic]           # shape (n_sites, n_haplotypes)

# 2. Get reference and alternative alleles
variants = list(ts.variants())
ref = np.array([v.alleles[0] for v in variants])[biallelic]
alt = np.array([v.alleles[1] for v in variants])[biallelic]

# 3. Simulate allele read counts
arc = simGL.sim_allelereadcounts(
    gm, mean_depth=10., std_depth=2., e=0.01,
    ploidy=2, seed=42, ref=ref, alt=alt,
)
# arc shape: (n_sites, n_individuals, 4)  —  A, C, G, T read counts

# 4. Compute genotype likelihoods
GL = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2)
# GL shape: (n_sites, n_individuals, 10)  —  all diploid ACGT genotypes

# 5. Subset to biallelic genotypes and write a VCF
Ra     = simGL.ref_alt_to_index(ref, alt)
GL_sub = simGL.subset_GL(GL, Ra, ploidy=2)

pos   = np.array([int(v.site.position) for v in variants])[biallelic] + 1
names = [f"ind{i}" for i in range(ts.num_individuals)]
simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos, names, "output.vcf")
```

## Documentation

Full documentation — installation, user guide, API reference, and theory — is
available at **https://simgl.readthedocs.io**.

## Citation

If you use simGL in your work, please cite the relevant methodological papers
listed in the [Citation page](https://simgl.readthedocs.io/en/latest/citation.html)
of the documentation.

## License

[ISC](LICENSE)
