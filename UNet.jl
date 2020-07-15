#=
upsample : ConvTranspose((2,2), ifilters=>ifilters/2, relu, stride=(2,2))
 ou upsample(x, ratio) puis Conv((2,2), ifilters=>ifilters/2) problem padding
 Conv((1,1))
=#

# https://github.com/FluxML/NNlib.jl/pull/95

# Pb avec GPU ou Zygote???
function upsample(x, ratio) #upsamle(img, (2,2,1,1)) x -> upsample(x, (2,2,1,1))
  y = similar(x, (size(x) .* ratio)...)
  for i in Iterators.product(Base.OneTo.(ratio)...)
     loc = map((i,r,s)->range(i, stop = s, step = r), i, ratio, size(y))
     @inbounds y[loc...] = x
  end
  return y
end

upsample(x) = upsample(x, (2,2,1,1))

using Flux
using Flux: @functor #, convfilter, Zeros
using Distributions: Normal

include("convwb.jl")

#init_ones(shape...) = ones(Float32, shape...)

#ifunc = glorot_uniform

# Params : inchannnels, nclasses, activation, init, batchnorm

function convbn(kernel, filters, activation, batchnorm, init, pad=(1,1))
  ifilters, ofilters = filters
  layers = []
  if batchnorm
    push!(layers, ConvWB(kernel, filters, pad=pad, init=init))
    push!(layers, BatchNorm(ofilters, activation))
  else
    push!(layers, Conv(kernel, filters, activation, pad=pad, init=init))
  end
  return layers
end

function uconv(filters, activation, batchnorm, init)
  ifilters, ofilters = filters
  layers = []
  push!(layers, convbn((3,3), ifilters=>ofilters, activation, batchnorm, init)...)
  push!(layers, convbn((3,3), ofilters=>ofilters, activation, batchnorm, init)...)
  return layers
end

struct UBlock
  up
  conv
end

@functor UBlock

function UBlock(filters, activation, batchnorm, init)
  ifilters, ofilters = filters
  mfilters = Int(ifilters / 2)
  up = ConvTranspose((2,2), ifilters=>mfilters, activation, stride=(2,2), init=init)
  conv = Chain(uconv(filters, activation, batchnorm, init)...)
  return UBlock(up, conv)
end

(d::UBlock)(x, bridge) = d.conv(cat(d.up(x), bridge, dims=3))

struct UNet
  enc1
  enc2
  enc3
  enc4
  enc5
  dec1
  dec2
  dec3
  dec4
  output
end

@functor UNet

function UNet(;inchannels, activation=relu, batchnorm=false, init=glorot_uniform) # ajout du nombre de classes
  enc1 = Chain(uconv(inchannels=>64, activation, batchnorm, init)...)
  enc2 = Chain(MaxPool((2,2)), uconv(64=>128, activation, batchnorm, init)...)
  enc3 = Chain(MaxPool((2,2)), uconv(128=>256, activation, batchnorm, init)...)
  enc4 = Chain(MaxPool((2,2)), uconv(256=>512, activation, batchnorm, init)...)
  enc5 = Chain(MaxPool((2,2)), uconv(512=>1024, activation, batchnorm, init)...)
  dec1 = UBlock(1024=>512, activation, batchnorm, init)
  dec2 = UBlock(512=>256, activation, batchnorm, init)
  dec3 = UBlock(256=>128, activation, batchnorm, init)
  dec4 = UBlock(128=>64, activation, batchnorm, init)
  output = Chain(convbn((1,1), 64=>1, sigmoid, batchnorm, init, (0,0))...)
  return UNet(enc1, enc2, enc3, enc4, enc5, dec1, dec2, dec3, dec4, output)
end

function (u::UNet)(x)
  b1 = u.enc1(x)
  b2 = u.enc2(b1)
  b3 = u.enc3(b2)
  b4 = u.enc4(b3)
  b5 = u.enc5(b4)
  d1 = u.dec1(b5, b4)
  d2 = u.dec2(d1, b3)
  d3 = u.dec3(d2, b2)
  d4 = u.dec4(d3, b1)
  return u.output(d4)
end
