function CPM(mat;M=1e6)
    return mat ./sum(mat,dims=2)[:,1].*M
end

function delog_base(mat; base=2)
    clamp.(base .^ (mat) .- 1, 0, Inf)
end

function gradient_step_LINCS(X, Q, model, opt, loss)
    ps=Flux.params(model) 
    grads = Flux.gradient(ps) do
        loss(model, X, Q)
    end

    Flux.update!(opt, ps, grads)
    train_l_tmp = loss(model, X, Q)

    return train_l_tmp
end

#! Kinda unoptimized
function l2_penalty(model)
    sum_w=0
    for param in Flux.params(model)
        sum_w+=sum(abs2,param)
    end
    return sum_w
end

# Extra agressive MSE. Not needed, probably
function mqe(ŷ, y; agg = mean)
    #_check_sizes(ŷ, y)
    agg(abs.(ŷ .- y).^4)
end


function train_fold_LINCS(data_x, data_y; model_type="skip-ae", 
    loss_name="MSE", n_epochs=10000, test_size=35, batch_size=20,
    lr = 1e-4, l2=1e-4)

    n_samples = size(data_x)[1]
    in_dim = size(data_y)[2]

    shuffled_ids = shuffle(collect(1:n_samples))
    
    test_ids = shuffled_ids[1:test_size]
    train_ids = shuffled_ids[test_size+1:end]

    n_batches = Int(ceil(length(train_ids)/batch_size))

    opt = ADAM(lr) #1e-4
    loss(model, x, y) = Flux.mse(model(gpu(x')), gpu(y'))  + l2*l2_penalty(model)
    # Alternative versions
    #loss(model, x, y) = Flux.mse(model(gpu(x')), gpu(y'), agg=x->mean(sum(x, dims=1)))  + l2*l2_penalty(model)
    #loss(model, x, y) = mqe(model(gpu(x')), gpu(y'), agg=x->mean(sum(x, dims=1))) # + L2 regularisation

    #model = gpu(Flux.Dense(in_dim, in_dim))
    model = gpu(SkipConnection(
        Flux.Chain(Flux.Dense(in_dim, 2000,relu),
            SkipConnection(
                #Flux.Chain(Flux.Dense(2000,2000,relu),
                Flux.Dense(2000,2000,relu)
                #)
            ,+),
            Flux.Dense(2000,in_dim)),
        +))

    test_x  = data_x[test_ids, :]
    test_y  = data_y[test_ids, :]
    
    train_loss = []
    test_loss  = []
    l2_loss = []
    #! There should be a seperate train_loop function 
    for e in 1:n_epochs
        epoch_shuffle=shuffle(train_ids)

        for b in 1:n_batches
            # Batch idx
            if b == n_batches
                batch_idx= epoch_shuffle[(b-1)*batch_size+1:end]
            else
                batch_idx = epoch_shuffle[(b-1)*batch_size+1:b*batch_size]
            end   

            batch_x = data_x[batch_idx, :]
            batch_y = data_y[batch_idx, :]

            train_l_tmp = gradient_step_LINCS(batch_x, batch_y, model, opt,loss)
            test_l_tmp = loss(model, test_x, test_y)

            push!(train_loss, train_l_tmp)
            push!(test_loss, test_l_tmp)
            push!(l2_loss, cpu(l2*l2_penalty(model)))

        end
        if (e % 10 == 0) || (e == 1) || (e == n_epochs) 
            println("Epoch: ", e, "   Train : ", train_loss[end] , " Test : ", test_loss[end])
        end
    end
    #! Save the results in a folder, should be better.
    return model, train_loss, test_loss, train_ids, test_ids, l2_loss

end

