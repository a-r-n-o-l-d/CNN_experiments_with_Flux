mutable struct Metrics
  accuracy
  loss
  epoch
  abuff
  lbuff
  afunc
  lfunc
  function Metrics(afunc, lfunc)
    a::Vector{Float64} = []
    l::Vector{Float64} = []
    ab::Vector{Float64} = []
    lb::Vector{Float64} = []
    new(a, l, 0, ab, lb, afunc, lfunc)
  end
end

function update!(metrics, model, x, y)
  testmode!(model)
  ȳ = model(x)
  push!(metrics.abuff, metrics.afunc(ȳ, y))
  push!(metrics.lbuff, metrics.lfunc(ȳ, y))
end

function update!(metrics)
  push!(metrics.accuracy, mean(metrics.abuff))
  push!(metrics.loss, mean(metrics.lbuff))
  resize!(metrics.abuff, 0)
  resize!(metrics.lbuff, 0)
  metrics.epoch = metrics.epoch + 1
end

function Base.show(io::IO, metrics::Metrics)
  if metrics.epoch > 0
    println(io, "Current epoch: $(metrics.epoch)")
    println(io, "\t accuracy: $(metrics.accuracy[end])")
    println(io, "\t loss: $(metrics.loss[end])")
  end
end
