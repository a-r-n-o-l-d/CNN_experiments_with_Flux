#=
upsample : ConvTranspose((2,2), ifilters=>ifilters/2, relu, stride=(2,2))
 ou upsample(x, ratio) puis Conv((2,2), ifilters=>ifilters/2) problem padding
 Conv((1,1))


# https://github.com/FluxML/NNlib.jl/pull/95
function upsample(x, ratio) #upsamle(img, (2,2,1,1)) x -> upsample(x, (2,2,1,1))
  y = similar(x, (size(x) .* ratio)...)
  for i in Iterators.product(Base.OneTo.(ratio)...)
     loc = map((i,r,s)->range(i, stop = s, step = r), i, ratio, size(y))
     @inbounds y[loc...] = x
  end
  return y
end
=#

using Flux
using Flux: @functor

# ajout de batchnorm
function conv3x3(filters; downsample)
  ifilters, ofilters = filters
  layers = []
  if downsample
    push!(layers, MaxPool((2,2)))
  end
  push!(layers, Conv((3,3), ifilters=>ofilters, relu, pad=(1,1)))
  push!(layers, Conv((3,3), ofilters=>ofilters, relu, pad=(1,1)))
  return layers
end

conv3x3(filters) = conv3x3(filters, downsample=false)

struct DecoderBlock
  upsample
  conv
end

@functor DecoderBlock

function DecoderBlock(filters)
  ifilters, ofilters = filters
  mfilters = Int(ifilters / 2)
  upsample = ConvTranspose((2,2), ifilters=>mfilters, stride=(2,2))
  conv = Chain(conv3x3(filters)...)
  return DecoderBlock(upsample, conv)
end

(d::DecoderBlock)(x, bridge) = d.conv(cat(d.upsample(x), bridge, dims=3))

struct Unet
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

@functor Unet

function Unet(;inchannels=3) # ajout du nombre de classes
  enc1 = Chain(conv3x3(inchannels=>64)...)
  enc2 = Chain(conv3x3(64=>128, downsample=true)...)
  enc3 = Chain(conv3x3(128=>256, downsample=true)...)
  enc4 = Chain(conv3x3(256=>512, downsample=true)...)
  enc5 = Chain(conv3x3(512=>1024, downsample=true)...)
  dec1 = DecoderBlock(1024=>512)
  dec2 = DecoderBlock(512=>256)
  dec3 = DecoderBlock(256=>128)
  dec4 = DecoderBlock(128=>64)
  output = Conv((1,1), 64=>1, sigmoid)
  return Unet(enc1, enc2, enc3, enc4, enc5, dec1, dec2, dec3, dec4, output)
end

function (u::Unet)(x)
  b1 = u.enc1(x)
  b2 = u.enc2(b1)
  b3 = u.enc3(b2)
  b4 = u.enc4(b3)
  b5 = u.enc5(b4)
  d = u.dec1(b5, b4)
  d = u.dec2(d, b3)
  d = u.dec3(d, b2)
  d = u.dec4(d, b1)
  return u.output(d)
end
