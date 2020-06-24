using Augmentor
using Images

struct Augment
  scale
  noise
  pnoise
  pipeline
  insize
  outsize
  function Augment(; imsize, scale=1, noise=0, pnoise=0.5, level=3)
    p = makepipeline(imsize..., scale, level)
    o = map(x -> scale * x, imsize)
    new(scale, noise, pnoise, p, imsize, o)
  end
end

function (aug::Augment)(img)
  img = augment(img, aug.pipeline) |> normalize
  if aug.noise > 0
  if rand(1) < [aug.pnoise]
    wn = randn(eltype(img), size(img))
    return @. img + wn * aug.noise
    end
  end
  return img
end

function makepipeline(height, width, scale, level)
  nheight = scale * height
  nwidth = scale * width
  pipeline = NoOp() #Resize(nheight, nwidth)
  if level >= 1
    pipeline = FlipX(0.5) |> pipeline
  end
  if level >= 2
    pipeline = Zoom(0.9:0.01:1.3) |> pipeline # |> CropSize(height, width)
  end
  if level >= 3
    pipeline = Rotate(-10:1:10) |> CropSize(nheight, nwidth) |> pipeline
  end
  return Resize(nheight, nwidth) |> pipeline
end

function normalize(img) # [0,1] to [-1,1] like tensorflow
  img = permutedims(channelview(img), (3, 2, 1))[:,:,:,:]
  return @. (Float32(img) - 0.5) * 2
end
