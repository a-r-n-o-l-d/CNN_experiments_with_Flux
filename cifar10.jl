using MLDatasets

# création tableau de tuple (X, Y) X : images RGB, Y : onehotbatch
# tensors from test dataset
function testminibatches(hpar)
  outsize = hpar.augmentor.outsize
  datas = []
  n = length(CIFAR10.testlabels())
  for idx in partition(1:n, hpar.batchsize)
    n = length(idx)
    imgs = CIFAR10.convert2image(CIFAR10.testtensor(idx))
    X_batch = Array{Float32}(undef, outsize..., 3, n)
    Threads.@threads for i in 1:n
      img = augment(imgs[:, :, i], Resize(outsize))
      X_batch[:, :, :, i] = normalize(img)
    end
    labs = CIFAR10.testlabels(idx)
    Y_batch = onehotbatch(labs, 0:9)
    push!(datas, (X_batch, Y_batch))
  end
  return datas
end

function makeminibatch(idx, hpar)
  n = length(idx)
  imgs = CIFAR10.convert2image(CIFAR10.traintensor(idx))
  X_batch = Array{Float32}(undef, hpar.augmentor.outsize..., 3, n)
  Threads.@threads for i in 1:n
    X_batch[:, :, :, i] = hpar.augmentor(imgs[:, :, i])
  end
  labs = CIFAR10.trainlabels(idx)
  Y_batch = onehotbatch(labs, 0:9)
  if hpar.labelsmoothing > 0 # smoothing labels
    α = hpar.labelsmoothing
    Y_batch = (1 - α) .* Y_batch .+ α .* fill(0.1, size(Y_batch))
  end
  return (X_batch, Y_batch)
end
