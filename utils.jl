struct HyperParameters
  augmentor
  labelsmoothing
  batchsize
  epochs
  optimizer
  dropout
  fcsize
  function HyperParameters(; augmentor, labelsmoothing, batchsize, epochs, optimizer, dropout, fcsize)
    new(augmentor, labelsmoothing, batchsize, epochs, optimizer, dropout, fcsize)
  end
end

# Retrieve model parameters
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

# Check for NaN
anynan(x) = any(isnan.(x))
