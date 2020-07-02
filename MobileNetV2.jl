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
using Flux: convfilter, Zeros
using NNlib: relu6

function conv(ifilters, ofilters, kernel, stride, groups)
  if groups == 1
    c = Conv
  else
    c = DepthwiseConv
  end
  p = floor(Int, (kernel - 1) / 2)
  w = convfilter((kernel,kernel), ifilters=>ofilters)
  b = Zeros()
  layers = []
  push!(layers, c(weight=w, bias=b, pad=(p,p), stride=stride))
  push!(layers, BatchNorm(ofilters, relu6))
  return layers
end

function bottleneck(ifilters, ofilters, stride, expansion_factor)
  mfilters = round(Int, ifilters * expansion_factor)
  layers = []
  if expansion_factor != 1
    push!(layers, conv(ifilters, mfilters, 1, 1, 1)...)
  end
  push!(layers, conv(mfilters, mfilters, 3, stride, mfilters)...)
  w = convfilter((1,1), mfilters=>ofilters)
  b = Zeros()
  push!(layers, Conv(weight=w, bias=b, stride=(1,1), pad=(0,0)))
  push!(layers, BatchNorm(ofilters))
  if stride == 1 && ifilters == ofilters
    layers = SkipConnection(Chain(layers...), +)
  end
  return layers
end

function block()

end
