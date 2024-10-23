"""
Written by Safia Safa-tahar-henni and Sébastien Lemieux
"""

module Lincs

using HDF5, CSV, DataFrames
using ProgressBars

export Data, StrIndex, getindex, length
export compound, gene

## Provide a 2-way indexing between string and int

struct StrIndex
    str2id::Dict{String, Int32}
    id2str::Vector{String}

    StrIndex(vs::Vector{String}) = new(Dict(vs[i] => i for i = 1:length(vs)), vs) # must be uniqued!
    StrIndex(ds::HDF5.Dataset) = StrIndex(ds[:])
end

## Indexing

Base.getindex(idx::StrIndex, s::String) = idx.str2id[s]
Base.getindex(idx::StrIndex, i::Integer) = idx.id2str[i]
Base.getindex(idx::StrIndex, v::AbstractVector{String}) = [idx[s] for s in v]
Base.getindex(idx::StrIndex, v::AbstractVector{<:Integer}) = [idx[i] for i in v]
Base.getindex(idx::StrIndex, df::AbstractDataFrame) = mapcols(col -> idx[col], df)
Base.length(idx::StrIndex) = length(idx.id2str)

## HDF5 IO

Base.setindex!(f::HDF5.File, s::StrIndex, k::String) = setindex!(f, s.id2str, k)

function Base.setindex!(f::HDF5.File, df::AbstractDataFrame, k::String)
    g = create_group(f, k)
    for (name, vec) in pairs(eachcol(df))
        g[String(name)] = vec
    end
end

function DataFrames.DataFrame(g::HDF5.Group)
    convert(p) = (p.first, p.second[:]) # To pull data from the HDF5 dataset
    return DataFrame(Dict(map(convert, pairs(g))))
end

## LINCS data structure

struct Data
    expr::Matrix{Float32}
    gene_df::DataFrame
    gene_si::StrIndex ## Used as an index to _df
    compound_df::DataFrame
    compound_si::StrIndex ## Used as an index to _df
    inst_df::DataFrame ## Converted to identifiers only, use inst_si to convert back
    inst_si::StrIndex
end


function Data(fn::String)
    f = h5open(fn)
    gene_df = DataFrame(f["gene_df"])
    gene_si = StrIndex(f["gene_si"])
    #gene_si = nothing
    compound_df = DataFrame(f["compound_df"])
    #compound_df = nothing
    compound_si = StrIndex(f["compound_si"])
    #compound_si = nothing
    inst_df = DataFrame(f["inst_df"])
    inst_si = StrIndex(f["inst_si"])
    expr = f["expr"][:,:]
    #return Data(expr, gene_df, inst_df, inst_si)
    return Data(expr, gene_df, gene_si, compound_df, compound_si, inst_df, inst_si)
end

Base.getindex(d::Data, sym::Symbol) = d.inst_df[!, sym]
Base.getindex(d::Data, sym::Symbol, value::Int32) = d.inst_df[!, sym] .== value
Base.getindex(df::DataFrame, sym::Symbol, value::Int32) = df[!, sym] .== value
Base.getindex(d::Data, sym::Symbol, value::String) = d[sym, d.inst_si[value]]

Base.getindex(d::Data, sym::Symbol, v::AbstractVector{<:Integer}) = reduce(.|, [d[sym, value] for value = v])
Base.getindex(d::Data, sym::Symbol, v::AbstractVector{String}) = d[sym, d.inst_si[v]]
Base.getindex(d::Data, v::BitVector) = d.inst_df[v,:]
Base.getindex(d::Data, v::BitVector, z) = d.inst_df[v,z]

Base.unique(d::Data, sym::Symbol) = unique(d.inst_df[!, sym])
(d::Data)(value::Integer) = d.inst_si[value]
(d::Data)(v::AbstractVector{<:Integer}) = d.inst_si[v]

compound(d::Data) = d.compound_df
compound(d::Data, id::String) = d.compound_df[d.compound_si[id],:]

gene(d::Data) = d.gene_df
gene(d::Data, sym::String) = d.gene_df[d.gene_si[sym], :]

function Data(prefix::String, gctx::String, out_fn::String; feature_space="landmark")
    f_out = h5open(out_fn, "w")

    println("Loading from original files...")
    f = h5open(prefix * gctx)
    expr = f["0/DATA/0/matrix"]
    exprGene_idx = StrIndex(f["0/META/ROW/id"][:])
    
    ## Compound information

    println("Loading compound annotations...")
    compound_df = CSV.File(prefix * "compoundinfo_beta.txt", 
                           delim="\t", types=String, missingstring=nothing, pool=false) |> DataFrame
    gdf = groupby(compound_df, [:pert_id, :canonical_smiles, :inchi_key])
    # pert_id unique per smiles/key. some have multiple cmap_name, keep only the first
    
    compound_df = combine(gdf, :cmap_name => (x -> first(x)) => :first_name)
    compound_si = StrIndex(compound_df.pert_id)

    ## Gene and sample annotations
    
    println("Loading gene and sample annotations...")
    gene_df = CSV.File(prefix * "geneinfo_beta.txt",
                       delim="\t", types=String, missingstring=nothing, pool=false) |> DataFrame
    dfGene_idx = StrIndex(gene_df[:,"gene_id"])
    
    inst_df = CSV.File(prefix * "instinfo_beta.txt", delim = '\t', types=String, missingstring=nothing, pool=false) |> DataFrame
    dfInst_idx = StrIndex(inst_df[:,"sample_id"])
    exprInst_idx = StrIndex(f["0/META/COL/id"][:])
    exprInst_o = dfInst_idx[exprInst_idx.id2str]
    inst_df = inst_df[exprInst_o,:] ## Reorder the df to fit the matrix
    
    println("Preparing sample annotation global StrIndex...")
    inst_si = StrIndex(unique(reduce(vcat, [unique(c) for c in eachcol(inst_df)])))
    for i in names(inst_df)
        inst_df[!, i] = inst_si[inst_df[!,i]]
    end

    println("Subsetting landmark genes")
    
    
    g = groupby(gene_df, "feature_space")
    
    ##!# Find a way to get it all
    #lm_id = get(g, (feature_space="landmark",), nothing).gene_id
    lm_id = gene_df.gene_id
    gene_df = gene_df[dfGene_idx[lm_id],:]
    lm_sym = StrIndex(gene_df.gene_symbol)
    #lm_row = exprGene_idx ## convert to the matrix rows
    lm_row = exprGene_idx[lm_id] ## convert to the matrix rows
    
    z = zeros(Float32, (size(expr)[1], length(lm_row)))
    for i=1:length(lm_row)
        z[lm_row[i],i] = 1
    end

    chunk_size = 8 * 4096
    ngene, ninst = size(expr)

    nlm = length(lm_row)

    final = Matrix{Float32}(undef, (nlm, ninst)) 

    ## This actually loads the file (about 15 minutes)
    for start in ProgressBar(1:chunk_size:ninst)
        r = start:min(start+chunk_size-1, ninst)
        final[:, r] = z' * expr[:,r]
    end
    
    println("Saving parsed dataset to $(out_fn)")
    
    # About 1GB for all indices and dataframes
    f_out["gene_df"] = gene_df
    f_out["gene_si"] = lm_sym
    f_out["compound_df"] = compound_df
    f_out["compound_si"] = compound_si
    f_out["inst_df"] = inst_df
    f_out["inst_si"] = inst_si
    
    # About 11GB for the expressions (landmark genes only)
    f_out["expr"] = final
    close(f_out)
    
    return Data(final, gene_df, lm_sym, compound_df, compound_si, inst_df, inst_si)
end

end