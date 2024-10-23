"""
The code used to generate pseudo-RNA-seq counts. The data is unavailable in the Git Repository due to size limitations
"""

import Pkg
Pkg.activate("src/.")
include("utils_DDPM.jl")
include("conditional_mlp.jl")
include("samplings.jl")
include("ML_utils_LINCS.jl")
using CSV, DataFrames, StatsBase
push!(LOAD_PATH, "SETUP")
include("Lincs.jl")
using .Lincs: Data
using .Lincs: StrIndex


function create_filter(lm_data, criteria::Dict{Symbol, String})
    filters = []
    for (k,v) in criteria
        f = lm_data[k, v]
        push!(filters, f)
    end
    return reduce(.&, filters)
end

# The counts and profile information for CCLE
df_CCLE_count = CSV.File("/home/golem/scratch/munozc/DDPM/Data/OmicsExpressionGenesExpectedCountProfile.csv",types = String, missingstring=nothing, pool=false) |> DataFrame
df_model = CSV.File("/home/golem/scratch/munozc/DDPM/Data/OmicsProfiles.csv",types = String, missingstring=nothing, pool=false) |> DataFrame

nam = names(df_CCLE_count)
nam[1] = "ProfileID"
rename!(df_CCLE_count, [i[1] for i in split.(nam, " ")], makeunique=true)

nam2 = names(df_model)
nam2[3] = "DepMap_ID" 
rename!(df_model, nam2)

df_all = innerjoin(df_model, df_CCLE_count, on=:ProfileID)

# Previously filtered datset, allows for easy filtration of some genes and cell lines
df_CCLE = CSV.File("/u/safasa/LINCS/Data/rnaseq/cancerCL/CCLE_expression_2.csv",types = String, missingstring=nothing, pool=false) |> DataFrame
mask = df_CCLE[:,"TSPAN6"] .!= "NA"
df_CCLE = df_CCLE[mask,:]

# Complete L1000 matrix
@time lm_data = Data("/home/golem/scratch/munozc/DDPM/Data/out_full.h5")


ctrl_criteria = Dict{Symbol, String}(:qc_pass => "1")

ctrl_filter = create_filter(lm_data, ctrl_criteria)
# codes for the correct pertubation types
v=  [3173106,3173107,3173108,3173109]
ctrl_filter = reduce(.|, [lm_data[:pert_type, lm_data.inst_si[value]] for value = v]) .& ctrl_filter
ctrl_filtered_inst_df = lm_data[ctrl_filter]


ctrl_cell_lines = unique(ctrl_filtered_inst_df[!,:cell_iname])

ctrl_cell_lines
Lincs_cell = ctrl_filtered_inst_df[!,:cell_iname]

M_lincs = lm_data.expr[:,ctrl_filter]

# Common cell lines
CCLE_cell = lm_data.inst_si[df_CCLE[!,:stripped_cell_line_name]]
common_sym = unique(intersect(names(df_CCLE),  lm_data.gene_df.gene_symbol, names(df_all)))
common_sym = filter(x->x!="MIA2",common_sym)

CCLE_ld = df_CCLE[!,common_sym]


CCLE_ld_count = deepcopy(CCLE_ld)

M_CCLE = parse.(Float32,CCLE_ld) |> Matrix{Float32}
Mt_CCLE = transpose(M_CCLE)

lg_order = [findfirst(sym .== lm_data.gene_df.gene_symbol) for sym in common_sym]
lm_data.gene_df
M_lincs = M_lincs[lg_order, :]

which_cell = [i == j ? Float32(1) : Float32(0) for i in Lincs_cell, j in CCLE_cell]
cellcount_coef = 1 ./ sum(which_cell, dims=1)


percell_avg = M_lincs * which_cell .* cellcount_coef
n_gene, n_cell = size(percell_avg)

# Replace values in count matrix
for i in ProgressBar(df_CCLE.DepMap_ID)
    for j in names(CCLE_ld)
        #println(i,j)
        @inbounds CCLE_ld_count[df_CCLE.DepMap_ID .== i, j] = string.((df_all[df_all.DepMap_ID .== i,j]))
    end
end

M_CCLE = parse.(Float32,CCLE_ld_count) |> Matrix{Float64}


Mt_CCLE = transpose(M_CCLE)

#!# Post-counts
Mt_CCLE_CPM = M_CCLE ./sum(M_CCLE,dims=2)[:,1].*1e6

#!# Post_CPM log
Mt_CCLE_CPM_log = log2.(Array(Mt_CCLE_CPM).+1)


"""
Methodology recap:
Gor from CCLE to pseudo-CPM for L1000 for z-scores.
Apply z-scores to our L1000 data.
Use as probabilities as CP1. 
Apply sampling on it for given coverage.  
"""

M_lincs # L100 sans moyenne pre-rescale
M_CCLE # CCLE (Log-tpm)
percell_avg # La moyenne sur les lignées L1000 sans rescale

"""
Z-swap function

Requires n_samples x n_features 
"""
function z_swap(mat1, mat2)
    swapped = (mat1 .- mean(mat1, dims=1))./std(mat1,dims=1).*std(mat2 ,dims=1) .+ mean(mat2 ,dims=1)
    return swapped
end

Mt_lincs=Array(M_lincs')
Mt_lincs_remean = z_swap(Mt_lincs, Mt_CCLE_CPM_log)

Mt_lincs_delog = delog_base(Mt_lincs_remean, base=2)
Mt_lincs_probs = Mt_lincs_delog ./sum(Mt_lincs_delog,dims=2)[:,1]

#!# TODO: avg on sample, not probability. That is much slower, but should lead to about the same. 
precell_avg = Array((Mt_lincs_probs' * which_cell .* cellcount_coef)')
Mt_CCLE_probs = M_CCLE ./sum(M_CCLE,dims=2)[:,1]


# To sample given a coverage 

coverage = 2e6
CCLE_remean = log2.([i[1] for i in rand.(Distributions.Binomial.(coverage, Mt_CCLE_probs[:,:]), 1)].+1)
precell_remean = log2.([i[1] for i in rand.(Distributions.Binomial.(coverage, precell_avg[:,:]), 1)].+1)
