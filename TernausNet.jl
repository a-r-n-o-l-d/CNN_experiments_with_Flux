struct EncoderBlock
  downsample
end

@functor EncoderBlock

function EncoderBlock(ifilters, ofilters, depth)
  layers = []
  push!(layers, MaxPool((2,2)))
  for l in 1:depth
    push!(layers, Conv((3,3), ifilters=>ofilters, relu, pad=(1,1)))
    ifilters = ofilters
  end
  return EncoderBlock(Chain(layers...))
end

(e::EncoderBlock)(x) = e.downsample(x)

struct DecoderBlock
  upsample
  bridge
end

@functor DecoderBlock

function DecoderBlock(ifilters, mfilters, ofilters, bridge)
  conv = Conv((3,3), ifilters=>mfilters, relu, pad=(1,1))
  upconv = ConvTranspose((4,4), mfilters=>ofilters, relu, stride=(2,2), pad=(1,1)) #(4,4) stride(2,2) #https://github.com/ternaus/robot-surgery-segmentation
  return DecoderBlock(Chain(conv, upconv))
end

(d::DecoderBlock)(x, bridge) = cat(d.upsample(x), bridge, dims=3)
#function (d::DecoderBlock)(x) = return d.upsample(x)

struct TernausNet
  enc1    # 3=>64,             (w,h)
  enc2    # 64=>128,           (w/2,h/2)
  enc3    # 128=>256,          (w/4,h/4)
  enc4    # 256=>512,          (w/8,h/8)
  enc5    # 512=>512,          (w/16,h/16)
  center  # 512=>512,          (w/32,h/32) juste MaxPool((2,2))
  dec1    # 512=>512=>256+512, (w/16,h/16)
  dec2    # 768=>512=>256+512, (w/8,h/8)
  dec3    # 768=>512=>128+256, (w/4,h/4)
  dec4    # 384=>128=>64+128,  (w/2,h/2)
  dec5    # 192=>128=>32+64,   (w,h)
  output  # Conv((3,3), 96=>32, relu, pad=(1,1)) Conv((1,1), 32=>1, sigmoid, pad=(1,1)) oukernel (3,3)
end

@functor TernausNet

function (t::TernausNet)(x)
  d1 = t.enc1(x)
  d2 = t.enc2(d1)
  d3 = t.enc3(d2)
  d4 = t.enc4(d3)
  d5 = t.enc5(d4)
  c = t.center(d4)
  u1 = t.dec1(c, d5)
  u2 = t.dec2(u1, d4)
  u3 = t.dec3(u2, d3)
  u4 = t.dec4(u3, d2)
  u5 = t.dec5(u4, d1)
  return t.ouput(u5)
end
