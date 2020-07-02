function conv3x3(ifilters, ofilters, activation)
  [Conv(weight=convfilter((3,3), ifilters=>ofilters), bias=Zeros(), pad=(1,1)),
   BatchNorm(ofilters, activation)]
end

conv3x3(ifilters, ofilters) = conv3x3(ifilters, ofilters, identity)

function conv1x1(ifilters, ofilters, stride=(1,1))
  [Conv(weight=convfilter((1,1), ifilters=>ofilters), bias=Zeros(), pad=(1,1), stride=stride),
   BatchNorm(ofilters)]
end

struct BasicBlock
  conv
  downsample
end

@functor BasicBlock

function BasicBlock(ifilters, stride)

end

function (b::BasicBlock)(x) = relu.(b.conv(x) + b.downsample(x))
