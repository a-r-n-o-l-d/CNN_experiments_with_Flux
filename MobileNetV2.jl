#=
function weight_params(m::Chain, ps=Flux.Params())
    map((l)->weight_params(l, ps), m.layers)
    ps
end
weight_params(m::Dense, ps=Flux.Params()) = push!(ps, m.W)
weight_params(m::Conv, ps=Flux.Params()) = push!(ps, m.weight)
weight_params(m::ConvTranspose, ps=Flux.Params()) = push!(ps, m.weight)
weight_params(m, ps=Flux.Params()) = ps
=#
# https://arxiv.org/pdf/1801.04381.pdf
#NNlib.relu6
using Flux
using Flux: convfilter, depthwiseconvfilter, Zeros
using NNlib: relu6

const iresiduals = [# t, c,   n, s
                     (1, 16,  1, 1),
                     (6, 24,  2, 2),
                     (6, 32,  3, 2),
                     (6, 64,  4, 2),
                     (6, 96,  3, 1),
                     (6, 160, 3, 2),
                     (6, 320, 1, 1)]

function conv(ifilters, ofilters, kernel, stride, dw) #convbnr6
  if dw
    cnv = DepthwiseConv
    w = depthwiseconvfilter((kernel,kernel), ifilters=>ofilters)
  else
    cnv = Conv
    w = convfilter((kernel,kernel), ifilters=>ofilters)
  end
  p = floor(Int, (kernel - 1) / 2)
  b = Zeros()
  layers = []
  push!(layers, cnv(weight=w, bias=b, pad=(p,p), stride=stride))
  push!(layers, BatchNorm(ofilters, relu6))
  return layers
end

function bottleneck(ifilters, ofilters, stride, expansion)
  mfilters = round(Int, ifilters * expansion)
  layers = []
  if expansion != 1
    push!(layers, conv(ifilters, mfilters, 1, 1, false)...)
  end
  push!(layers, conv(mfilters, mfilters, 3, stride, true)...)
  w = convfilter((1,1), mfilters=>ofilters)
  b = Zeros()
  push!(layers, Conv(weight=w, bias=b, stride=(1,1), pad=(0,0)))
  push!(layers, BatchNorm(ofilters))
  if stride == 1 && ifilters == ofilters
    layers = [SkipConnection(Chain(layers...), +)]
  end
  return layers
end

function block(ifilters, ofilters, stride, expansion, repeat)
  layers = []
  push!(layers, bottleneck(ifilters, ofilters, stride, expansion)...)
  for i in 2:repeat
    push!(layers, bottleneck(ofilters, ofilters, 1, expansion)...)
  end
  return layers
end

function classifier(ifilters, nclasses, dropout)
  layers = []
  push!(layers, GlobalMeanPool())
  push!(layers, flatten)
  push!(layers, Dropout(dropout))
  push!(layers, Dense(ifilters, nclasses))
  push!(layers, softmax)
end

function makedivisible(x, divisor, minval=divisor)
  x_ = max(minval, floor(Int, floor(Int, x + divisor / 2) / divisor) * divisor)
  # Make sure that round down does not go down by more than 10%.
  if x_ < 0.9 * x
    x_ += divisor
  end
  return x_
end

function mobilenetv2(; inchannels=3, nclasses, wmult=1, rnearest=8, dropout=0.2)
  layers = []
  ifilters = makedivisible(32 * wmult, rnearest)
  push!(layers, conv(inchannels, ifilters, 3, 2, false)...)
  for r in iresiduals
    expansion, filters, repeat, stride = r
    ofilters = makedivisible(filters * wmult, rnearest)
    push!(layers, block(ifilters, ofilters, stride, expansion, repeat)...)
    ifilters = ofilters
  end
  c = makedivisible(1280 * wmult, rnearest)
  push!(layers, conv(ifilters, c, 1, 1, false)...)
  push!(layers, classifier(c, nclasses, dropout)...)
  return(Chain(layers...))
end
