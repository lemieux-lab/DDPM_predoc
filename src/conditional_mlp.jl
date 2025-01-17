include("time_embeddings.jl")


# Struct 1 : the conditional layer receiving the timsestep embedding
mutable struct Conditional_layer
    activation
    in_norm     :: Flux.LayerNorm
    in_fc       :: Flux.Dense
    out_norm    :: Flux.LayerNorm
    out_fc      :: Flux.Dense
    proj        :: Flux.Dense
    #dropout     :: Flux.Dropout
end


function Conditional_layer(in_dim, out_dim, temb_dim)

    activation = identity

    in_norm  = Flux.LayerNorm(in_dim, activation)

    in_fc    = Flux.Dense(in_dim => out_dim)
    out_norm = Flux.LayerNorm(out_dim, activation)

    out_fc   = Flux.Dense(out_dim => out_dim)
    proj     = Flux.Dense(temb_dim => out_dim)
    return Conditional_layer(activation, in_norm, in_fc, out_norm, out_fc, proj)
end


@Flux.functor(Conditional_layer)
function (cl::Conditional_layer)(x, t_emb)
    out = cl.in_fc(x)
    out = out .+ cl.proj(cl.activation(t_emb))
    out = cl.out_fc(cl.out_norm(out))
    return cl.activation(out)
end

# Strcut 2 : the conditional MLP itself
struct Conditional_MLP
    base_dim :: Int
    temb_dim :: Int

    drop :: Flux.Dropout
    
    embed :: Flux.Chain

    in_fc  :: Flux.Dense
    layers
    out_fc :: Flux.Dense
end



function Conditional_MLP(param_dict)
    """
    in_dim=1,
    base_dim=64,
    out_dim=None,
    multiplier=1,
    temb_dim=None,
    num_layers=3,
    drop_rate=0.,
    continuous_t=False
    """
    
    in_dim = param_dict["in_dim"]
    base_dim = param_dict["base_dim"]

    drop = Flux.Dropout(0.5)

    temb_dim = param_dict["base_dim"]
    out_dim = param_dict["in_dim"]

    num_layers = 1

    activation = identity 

    embed = Flux.Chain(Flux.Dense(base_dim => temb_dim, activation), Flux.Dense(temb_dim => temb_dim))

    in_sc = Flux.Scale(in_dim, )
    in_fc = Flux.Dense(in_dim => base_dim)
    test = [base_dim for i in 1:num_layers+1]
    layers = gpu.([Conditional_layer(test[r], test[r+1], temb_dim) for r in 1:num_layers])

    out_fc = Flux.Dense(base_dim => out_dim)

    return Conditional_MLP(base_dim, temb_dim, drop, embed, in_fc, layers, out_fc)
end


@Flux.functor(Conditional_MLP)
function (cl::Conditional_MLP)(x, t)
    t_emb = gpu(timestep_embedding(t,cl.base_dim))
    t_emb = cl.embed(t_emb')
    out = cl.in_fc(x) 

    for layer in cl.layers
        out = layer(out, t_emb)
    end
    out = leakyrelu.(out, 0.02)

    out = cl.out_fc(out)

    out = out .+ x
    return out

end
