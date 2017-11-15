function infer(M::Model, X)
  infer(M.alg, M.arc, X)
end

function update!(M::Model, y, ŷ, x, learn_rate::Float64)
    update!(M, M.alg, M.arc, y, ŷ, x, learn_rate)
end

"""
Linear Algorithm
"""
struct Linear <: Algorithm
    link
end

function Linear()::Linear
  Linear(identity)
end

function infer(alg::Linear, arc::SingleLayer, X)
  return arc.wts * X'
end

# function infer(alg::Linear, arc::MultiLayer, X)
# end

function update!(M::Model, alg::Linear, arc::SingleLayer,
                 y, ŷ, x, learn_rate::Float64)
  M.arc.wts .-= learn_rate * assessgrad(M, y, ŷ, x)
end

# function update!(M::Model, alg::Linear, arc::MultiLayer,
#                  y, ŷ, x, learn_rate::Float64)
# end
