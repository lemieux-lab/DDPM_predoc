# PoissonJUMP-DDPM-RNA

An implementation of a Poisson-JUMP DDPM inspired by (Chen \& Zhou, 2023) to denoise low-coverage RNA-seq data.

For julia version 1.10.3

## Usage

If you wish to use this code, it is recommended to open a Julia REPL in VScode in order to execute the functions and call them.

This code is still very much in developpement and unoptimized, but suggestions are appreciated!


## Files: DDPM results
diffusion_model_binom.jl : The min faile for the DDPM developpement

conditional_mlp.jl: Contains the model architectures

samplings.jl: Contains sampling methods, including binomial thinning

time_embeddings.jl: Contains the time embeddings

utils_DDPM.jl: Utility functions

## Files: L1000 results
get_counts.jl: File on how counts were obtained and downsampled from L1000.

Lincs.jl: File containing the data structure for L1000. Written by Sébastien Lemieux and Safia Safa-tahar-henni

ML_utils_LINCS.jl: utility functions


The data files are not available on this repo, but can be potentially transferred if requested