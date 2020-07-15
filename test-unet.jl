include("UNet.jl")
include("metrics.jl")

using ImageDraw
using FileIO
using Images
using Augmentor
using Statistics
using Flux: binarycrossentropy, params, Params #dice_coeff_loss, bce_loss, , logitbce_loss
using Base.Iterators: partition
using Random
using CUDAapi
if has_cuda()
  @info "CUDA is on"
  import CuArrays
  CuArrays.allowscalar(false)
else
  @info "CUDA is off"
end

function normalize(img)
  img = permutedims(channelview(img), (3, 2, 1))
  return Float32.(img) # .- 0.5) .* 200
end

# vérifier le resize sur ground truth toujours binaire
function loadimages(idx, imdirs, mkdirs)
  x = normalize(augment(load(imdirs[idx]), Resize((128,128))))
  y = round.(Float32.(augment(load(mkdirs[idx]), Resize((128,128)))))
  y = permutedims(y, (2, 1))[:,:,:] # verif manque permutedim
  return x, y
end

p = "/media/afertin/Stock1/CNN_exp/datasets/fish_image"
f = readdir(joinpath(p, "fish_01"));
m = replace.(f, "fish"=>"mask")
imgs = joinpath.(p, "fish_01", f);
masks = joinpath.(p, "mask_01", m);

function makeminibatch(idx, imgs, masks)
  pairs = [loadimages(i, imgs, masks) for i in idx]
  X = Flux.batch([pairs[i][1] for i in eachindex(pairs)])
  Y = Flux.batch([pairs[i][2] for i in eachindex(pairs)])
  return X, Y #!!!! Vérif dims de Y
end

function makeminibatchfake(idx, datas)
  pairs = datas[idx]
  X = Flux.batch([normalize(pairs[i][1]) for i in eachindex(pairs)]) #      [normalize(pairs[i][1]) for i in eachindex(pairs)])
  Y = Flux.batch([Float32.(permutedims(pairs[i][2], (2, 1)))[:,:,:] for i in eachindex(pairs)])
  return X, Y
end

function randomcircle()
  img = RGB.(rand(128, 128) ./ 2, rand(128, 128) ./ 2, rand(128, 128) ./ 2)
  o = round.((rand(2) .- 0.5) * 30) .+ 64
  r = round.((rand(1) .- 0.5) * 10)[1] + 20
  draw!(img, CirclePointRadius(Point(o...), r), RGB(1.0,1.0,1.0))
  mask = Gray.(zeros(128, 128))
  draw!(mask, CirclePointRadius(Point(o...), r), Gray(1.0))
  return img, mask
end

 model = UNet(inchannels=3, batchnorm=true) |> gpu
#opt = ADAM(1e-10)
#opt = ADAM(1e-10)
#opt = Descent(1e-3)
#opt = Flux.Optimiser(ExpDecay(0.1, 0.1, 1000, 1e-4), Descent())
opt = Descent()

#bce(ŷ, y) = mean(binarycrossentropy.(ŷ, y))
#loss(x, y) = dice_coeff_loss(model(x), y) #round. fout la merde
#loss(x, y) = bce_loss(model(x), y)
#loss(x, y) = logitbce_loss(model(x), y)

#=
function bce(ŷ, y; ϵ=gpu(fill(eps(first(ŷ)), size(ŷ)...)))
  l1 = -y.*log.(ŷ .+ ϵ)
  l2 = (1 .- y).*log.(1 .- ŷ .+ ϵ)
  l1 .- l2
end

function loss(x, y)
  op = clamp.(model(x), 0.001f0, 1.f0)
  mean(bce(op, y))
end
=#
#x = normalize(augment(load(imgs[1]), Resize((128,128)))) |> gpu
#Flux.train!(loss, Flux.params(model), gpu.([(im, gt)]), opt)

n = length(imgs)
#datas = [randomcircle() for _ in 1:n]

loss(x, y) = mean(Flux.binarycrossentropy.(model(x), y))
loss(x, y) = Flux.dice_coeff_loss(model(x), y)

xxx = []

trainmet = Metrics((ŷ, y) -> 1 - Flux.dice_coeff_loss(ŷ, y), (ŷ, y) -> mean(Flux.binarycrossentropy.(ŷ, y)))
for e in 1:20
  for idx in partition(shuffle(1:n), 32) #
    trainmode!(model)
    batch = makeminibatch(idx, imgs, masks) |> gpu #makeminibatchfake(idx, datas) |> gpu # #img, gt = loadimages(i, imgs, masks) |> gpu
    ps = Params(params(model))
    gs = gradient(ps) do
      training_loss = loss(batch...)
      #println(training_loss)
      return training_loss
    end
    Flux.Optimise.update!(opt, ps, gs)
    update!(trainmet, model, batch...)
  end
  update!(trainmet)
  @show trainmet
end

#=
function my_custom_train!(loss, ps, data, opt)
  ps = Flux.Params(ps)
  for d in data
    gs = gradient(ps) do
      training_loss = loss(d...)
      # Insert what ever code you want here that needs Training loss, e.g. logging
      return training_loss
    end
    # insert what ever code you want here that needs gradient
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
    update!(opt, ps, gs)
    # Here you might like to check validation set accuracy, and break out to do early stopping
  end
end

d = gpu.([(im, gt)])[1]
ps = Flux.params(model)
gs = gradient(ps) do
  training_loss = loss(d...)
  # Insert what ever code you want here that needs Training loss, e.g. logging
  return training_loss
end


=#
