using CUDAapi
if has_cuda()
  @info "CUDA is on"
  import CuArrays
  CuArrays.allowscalar(false)
else
  @info "CUDA is off"
end

using FluxModels
using Flux
using Flux: onehotbatch, onecold, crossentropy, Optimiser
using Base.Iterators: partition
using Random
using Statistics

include("utils.jl")
include("metrics.jl")
include("augment.jl")
include("cifar10.jl")

const ε = 1.0f-10

function train!(model, hpar)
  n = length(CIFAR10.trainlabels())
  test = testminibatches(hpar)

  # Regularization
  #l1(x) = sum(x.^2)
  #l2(x) = sum(abs.(x))
  #wd = 0.01 # weight decay

  loss(x, y) = crossentropy(model(x) .+ ε, y .+ ε) #+ wd * sum(l2, params(model))
  accuracy(ŷ, y) = mean(onecold(cpu(ŷ)) .== onecold(cpu(y)))
  testmet = Metrics(accuracy, crossentropy)
  trainmet = Metrics(accuracy, crossentropy)

  ps = params(model)
  @info("Training....")
  for epoch_idx in 1:hpar.epochs
    for idx in partition(shuffle(1:n), hpar.batchsize)
      batch = gpu.([makeminibatch(idx, hpar)])
      trainmode!(model)
      Flux.train!(loss, ps, batch, hpar.optimizer)
      update!(trainmet, model, batch[1]...)
    end
    na = anynan(paramvec(model))
    if na
      @info("Shit happens! Model contains NaN parameters.")
      break
    end
    for batch in test
      update!(testmet, model, gpu(batch)...)
    end
    update!(trainmet)
    update!(testmet)
    @show "Train metrics"
    @show trainmet
    @show "Test metrics"
    @show testmet
  end

  return trainmet, testmet
end

function quicktest()
  hpar = HyperParameters(batchsize=32, epochs=5, optimizer=Descent(),
                         augmentor=Augment(imsize=(32,32), noise=0.0, level=1),
                         labelsmoothing=0.05, dropout=0.5, fcsize=128)
  model = vgg11((32,32), nclasses=10, dropout=hpar.dropout, fcsize=hpar.fcsize) |> gpu
  tr, ts = train!(model, hpar)
end
