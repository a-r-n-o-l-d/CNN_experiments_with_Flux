include("UNet.jl")

p = "/media/afertin/Stock1/CNN_exp/datasets/fish_image/fish_01/"
imgs = joinpath.(p, readdir(p));

p = "/media/afertin/Stock1/CNN_exp/datasets/fish_image/mask_01/"
masks = joinpath.(p, readdir(p));

model = Unet(inchannels=3) |> gpu
opt = Momentum()

Flux.train!(loss, Flux.params(u), rep, opt);
