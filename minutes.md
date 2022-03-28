# simGL

This python package simulates Genotype Likelihoods (GL) out of a haplotypic allele count matrix typically obtained from simulations.

NOTE: `simGL` is still in development and might have some bugs. Please, report them in issues if you were to encounter some or have suggestions for improvement.

### Installation

git clone https://github.com/RacimoLab/simGL
cd simGL
pip install .
### Example



```python
import msprime
import numpy as np
import simGL
```

Then, using the former, we are going to obtain a `tree sequence` of the simulation from which we will extract the `haplotypic genotype matrix` and other relevant information below.


```python
ts = msprime.sim_ancestry(
    population_size=10_000,
    samples=10,
    sequence_length=100_000,
    recombination_rate=1.25e-8,
    random_seed=1234,
    record_provenance=False,
)
ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=5678)

ts
```





<div>
  <style>
    .tskit-table thead tr th {text-align: left;padding: 0.5em 0.5em;}
    .tskit-table tbody tr td {padding: 0.5em 0.5em;}
    .tskit-table tbody tr td:first-of-type {text-align: left;}
    .tskit-details-label {vertical-align: top; padding-right:5px;}
    .tskit-table-set {display: inline-flex;flex-wrap: wrap;margin: -12px 0 0 -12px;width: calc(100% + 12px);}
    .tskit-table-set-table {margin: 12px 0 0 12px;}
    details {display: inline-block;}
    summary {cursor: pointer; outline: 0; display: list-item;}
  </style>
  <div class="tskit-table-set">
    <div class="tskit-table-set-table">
      <table class="tskit-table">
        <thead>
          <tr>
            <th style="padding:0;line-height:21px;">
              <img style="height: 32px;display: inline-block;padding: 3px 5px 3px 0;" src="https://raw.githubusercontent.com/tskit-dev/administrative/main/tskit_logo.svg"/>
              <a target="_blank" href="https://tskit.dev/tskit/docs/latest/python-api.html#the-treesequence-class"> Tree Sequence </a>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Trees</td><td>109</td></tr>
          <tr><td>Sequence Length</td><td>100000.0</td></tr>
          <tr><td>Time Units</td><td>generations</td></tr>
          <tr><td>Sample Nodes</td><td>20</td></tr>
          <tr><td>Total Size</td><td>27.8 KiB</td></tr>
          <tr>
            <td>Metadata</td><td style="text-align: left;">No Metadata</td></tr>
        </tbody>
      </table>
    </div>
    <div class="tskit-table-set-table">
      <table class="tskit-table">
        <thead>
          <tr>
            <th style="line-height:21px;">Table</th>
            <th>Rows</th>
            <th>Size</th>
            <th>Has Metadata</th>
          </tr>
        </thead>
        <tbody>




We can extract the `genotype matrix` from the tree sequence. Note that by default, we are simulating diploid individuals and thus there are double number of haplotypes (20) than the number of individuals that we simulated (10). 


```python
gm = ts.genotype_matrix()
print(gm.shape)
gm
```

    (141, 20)





    array([[1, 0, 1, ..., 0, 1, 1],
           [0, 0, 1, ..., 0, 1, 0],
           [1, 0, 1, ..., 0, 1, 1],
           ...,
           [0, 0, 1, ..., 0, 0, 0],
           [1, 1, 1, ..., 1, 1, 1],
           [0, 0, 1, ..., 0, 0, 0]], dtype=int8)



In this case, we have 141 SNPs.

Then, we can extract the reference and alternative alleles.


```python
ref = np.array([v.site.ancestral_state for v in ts.variants()])
alt = np.array([v.site.mutations[0].derived_state for v in ts.variants()])

print(ref.shape)
print(ref[:10])
print(alt.shape)
print(alt[:10])
```

    (141,)
    ['T' 'A' 'G' 'A' 'C' 'G' 'C' 'G' 'G' 'A']
    (141,)
    ['C' 'T' 'T' 'C' 'G' 'T' 'T' 'T' 'T' 'G']


Then, we can simulate allele read counts (`arc`). We must decide the mean read depth per haplotype and standard deviation. This will determine the shape of a normal distribution from which mean coverage per haplotype will be sampled. Alternatively, an array with the coverage means per haplotypic chromosomes can also be inputted. Then, the number of reads per haplotype per site will be sampled from a poison distribution. Finally, the reads of each allele (A, C, G and T) will be sampled from a multinomial distribution in which the probability of sampling an error allele is `e/3` and sampling the correct allele is `1-e`. Finally, `ploidy` columns will be sum together to form the allele read counts for an individual.


```python
e      = 0.05
ploidy = 2
arc = simGL.sim_allelereadcounts(gm = gm, ref = ref, alt = alt, 
                           mean_depth = 15., std_depth = 3., e = e, 
                           ploidy = ploidy, seed = 1234)
print(arc.shape)
arc
```

    (141, 10, 4)





    array([[[ 1, 12,  0, 13],
            [ 0, 22,  0,  9],
            [ 0,  0,  0, 51],
            ...,
            [ 1, 10,  0,  4],
            [ 1,  7,  0, 16],
            [ 1, 25,  0,  1]],
    
           [[21,  0,  1,  0],
            [21,  0,  0, 16],
            [33,  1,  1,  0],
            ...,
            [ 8,  0,  0,  6],
            [23,  0,  2,  0],
            [12,  1,  0, 16]],
    
           [[ 0,  0, 21,  8],
            [ 0,  1,  6, 25],
            [ 1,  1, 42,  0],
            ...,
            [ 1,  0,  5,  9],
            [ 0,  0, 25, 13],
            [ 0,  0,  1, 23]],
    
           ...,
    
           [[ 0, 28,  0,  0],
            [31,  1,  1,  2],
            [ 0, 38,  0,  2],
            ...,
            [ 0, 14,  0,  0],
            [ 0, 34,  1,  2],
            [ 0, 27,  1,  1]],
    
           [[ 0,  0, 27,  2],
            [ 1,  0, 39,  1],
            [ 7,  0, 24,  2],
            ...,
            [ 1,  0, 24,  1],
            [15,  2, 14,  3],
            [ 1,  1, 27,  0]],
    
           [[ 1, 26,  0,  0],
            [35,  0,  0,  1],
            [ 1, 40,  1,  2],
            ...,
            [ 0, 15,  2,  0],
            [ 1, 32,  1,  0],
            [ 0, 22,  0,  0]]])



The output above matches the number of sites (first dimension) of the `genotype matrix`, has half the size of the second dimension (number of haplotypes are now number of individuals depending on ploidy) and has an additional dimension of size 4. Each value in this array correspond to the number of reads map to that particular site for a particular individual and the index of the value in the third dimension corresponds to each allele (in order: "A", "C", "G" and "T").

For example, the first individual is heterozygous T/C for the first site.


```python
print(ref[0], alt[0], gm[0, 0:2])
```

    T C [1 0]


Correspondingly, we find the majority of alleles counts in the second and fourth positions of the array (corresponding to C and T respectively) and we also see that an error has been simulated for the first individual for the first site.


```python
arc[0, 0]
```




    array([ 1, 12,  0, 13])



Furthermore, the coverage of this individual is close to the mean depth defined (15x per haplotypic chromosome).


```python
arc[0, 0].sum()
```




    26



Finally, from the allele read counts, we can apply a GL model.


```python
GL  = simGL.allelereadcounts_to_GL(arc, e = e, ploidy = ploidy)
print(GL.shape)
GL
```

    (141, 10, 10)





    array([[[ 80.13934448,  40.40754996,  80.81509992, ...,  84.18239575,
              40.40754996,  31.62272927],
            [104.38617073,  30.30566247, 104.38617073, ..., 104.38617073,
              74.08050826,  67.99870932],
            [206.19561466, 206.19561466, 206.19561466, ..., 206.19561466,
              34.46352733,   0.        ],
            ...,
            [ 43.09909035,  10.10188749,  43.77484579, ...,  47.14214162,
              33.6729583 ,  30.96993655],
            [ 73.40475282,  50.50943745,  74.08050826, ...,  77.44780409,
              23.57107081,  12.7589838 ],
            [ 97.03323043,  13.52659012,  97.70898587, ..., 101.0762817 ,
              97.70898587,  97.03323043]],
    
           [[  0.        ,  14.19086419,  10.82356836, ...,  80.86102536,
              81.53678079,  84.90407662],
            [ 39.68586908,  53.87673328,  53.87673328, ..., 124.58994571,
              70.71321243,  59.90112542],
            [  0.        ,  18.93263362,  18.93263362, ..., 129.37764057,
             130.05339601, 133.42069184],
            ...,
            [ 14.79773148,  20.20377498,  20.20377498, ...,  47.14214162,
              26.93836664,  22.88383401],
            [  0.        ,  15.54237507,   8.80778341, ...,  84.90407662,
              86.2555875 ,  92.99017916],
            [ 45.76766803,  50.50943745,  53.87673328, ...,  94.28428324,
              40.40754996,  29.59546295]],
    
           [[ 97.65157907,  97.65157907,  26.93836664, ...,  12.74750245,
               0.        ,  65.30716893],
            [104.38617073, 101.0188749 ,  84.18239575, ...,  80.12786312,
               0.        ,   3.30988903],
            [165.76510198, 163.07356159,  25.01443256, ...,   0.        ,
              28.38172839, 169.80815325],
            ...,
            [ 43.09909035,  43.77484579,  26.93836664, ...,  26.92688528,
               0.        ,  10.75468021],
            [127.95724154, 127.95724154,  43.77484579, ...,  26.88095984,
               0.        ,  75.39757506],
            [ 92.99017916,  92.99017916,  89.62288333, ...,  88.94712789,
              12.17507924,   0.        ]],
    
           ...,
    
           [[113.2054355 ,  18.92115226, 113.2054355 , ..., 113.2054355 ,
             113.2054355 , 113.2054355 ],
            [  0.        ,  17.58112274,  17.58112274, ..., 121.29153804,
             115.23270181, 117.24848677],
            [153.63594818,  25.67870664, 153.63594818, ..., 153.63594818,
             146.90135652, 145.54984564],
            ...,
            [ 56.60271775,   9.46057613,  56.60271775, ...,  56.60271775,
              56.60271775,  56.60271775],
            [137.46374311,  22.97568489, 134.09644728, ..., 133.42069184,
             127.36185562, 129.37764057],
            [109.16238423,  18.24539682, 105.7950884 , ..., 105.11933296,
             102.42779257, 105.11933296]],
    
           [[109.16238423, 109.16238423,  18.24539682, ...,   0.        ,
              11.51080516, 101.0762817 ],
            [153.63594818, 154.31170362,  22.98716625, ...,   0.        ,
              22.98716625, 153.63594818],
            [ 76.08481185,  80.81509992,   0.        , ...,   7.3529403 ,
              16.83647915,  96.30006819],
            ...,
            [ 92.99017916,  93.6659346 ,  12.85083468, ...,   0.        ,
              12.85083468,  92.99017916],
            [ 37.00581005,  40.40754996,   0.        , ...,  41.04886132,
              40.40754996,  85.52242527],
            [105.11933296, 102.42779257,  14.87810099, ...,   0.        ,
              18.24539682, 109.16238423]],
    
           [[101.0762817 ,  14.20234555, 101.75203713, ..., 105.11933296,
             105.11933296, 105.11933296],
            [  0.        ,  23.65144032,  23.65144032, ..., 141.50679437,
             138.13949854, 137.46374311],
            [157.67899945,  23.66292168, 154.98745905, ..., 157.67899945,
             151.62016322, 153.63594818],
            ...,
            [ 60.64576902,  10.13633157,  53.91117736, ...,  52.55966648,
              53.91117736,  60.64576902],
            [125.3345893 ,  18.25687818, 122.64304891, ..., 125.3345893 ,
             126.01034474, 129.37764057],
            [ 88.94712789,  14.86661963,  88.94712789, ...,  88.94712789,
              88.94712789,  88.94712789]]])



The output is given as the log(GL) normalized by substracting the value of the maximum likelihood genotype per site per individual. As you might noticed, the third dimension has a size of 10 since it is the number of possible genotypes for a diploid individual simulated here (in order: "AA", "AC", "AG", "AT", "CC", "CG", ..., "TT").

Thus, the first individual must have a value of 0 in the seventh value in this array.


```python
GL[0, 0]
```




    array([80.13934448, 40.40754996, 80.81509992, 37.04025413, 35.66578054,
           43.77484579,  0.        , 84.18239575, 40.40754996, 31.62272927])


