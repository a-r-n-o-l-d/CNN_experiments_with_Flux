using Flux
using Flux: convfilter, Zeros

function convbn(ifilters, ofilters)
  Chain(Conv(weight=convfilter((3,3), ifilters=>ofilters), bias=Zeros(), pad=(1,1)),
        BatchNorm(ofilters, relu))
end

convbn(filters) = convbn(filters, filters)

residual(filters) = SkipConnection(Chain(convbn(filters), convbn(filters)), +)

classifier(nclasses) = Chain(GlobalMaxPool(), flatten, Dense(512, nclasses), softmax)

maxpool() = MaxPool((2,2))

function ResNet9(inchannels, nclasses)
  Chain(convbn(inchannels, 64), convbn(64, 128), maxpool(), residual(128),
        convbn(128, 256), convbn(256, 512), maxpool(), residual(512), classifier(nclasses))
end
