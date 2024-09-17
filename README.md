# MaxComp
Scripts used to perform max-cut algorithm and compartment analysis. Please cite: https://doi.org/10.1101/2024.07.02.600897.
## Package requirements
- alabtools (https://github.com/alberlab/alabtools)
- pickle
- numpy
- scipy
- networkx
- cvxopt
- cvxpy
- matplotlib
- seaborn
- pandas
## How to run:
Models generated by the Integrated Genome Modeling Platform (IGM) (https://github.com/alberlab/igm) (saved as for example H1_igm-model.hss) should be placed under directory Model/ for further usage. The precalculated compartment file (saved as for example H1_compartments.npy) and speckle distance file (saved as for example H1_speckle_distance.npy) for H1-hESC should be saved under directory Model/.

Script used to perform the max-cut algorithm and generate single-cell compartment files for a population of structures:
```
python embedding.py <cell type> <chromosome index> <starting index> <ending index>
```
where "cell type" can be GM, H1 or HFF (GM for GM12878, H1 for H1-hESC and HFF for HFFc6), "chromosome index" should be ranging from 1 to 23 (1 to 22 for H1 or HFF), "starting index" and "ending index" represent indices of single-cell structures.

The following command line generates single-cell compartment results from structure 0 to structure 499 for chromosome 6 of H1-hESC:
```
python max_cut.py H1 6 0 500
```

Script used to perform compartment analysis:
```
python analysis.py <cell type> <chromosome index> <starting index> <ending index>
```
Similarly, we can run the analysis based on the compartment files generated by the max-cut algorithm:
```
python analysis.py H1 6 0 500
```
All results will be generated under the current directory.
