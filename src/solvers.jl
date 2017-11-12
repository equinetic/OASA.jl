function train!(M::Model, args...; nargs...)
    train!(M.sol, M, args...; nargs...)
end

"""
Gradient Descent
================

"""
mutable struct GradientDescent <: Solver
  state_err::Float64
  iter::Int64
  batch_size::Int64
end

function GradientDescent()::GradientDescent
  GradientDescent(0., 0, 0)
end

function train!(S::GradientDescent,
                M::Model,
                x::AbstractVecOrMat,
                y::AbstractVecOrMat;
                learn_rate::Float64 = 0.001,
                err_tol::Float64 = 0.01,
                max_iter::Int64 = 1000)
  M.sol.iter = 0
  M.sol.state_err = 2 * err_tol
  M.inf["errors"] = get(M.inf, "errors", Vector{Float64}())
  while M.sol.iter < max_iter && M.sol.state_err > err_tol
    ŷ = infer(M, x)
    M.sol.state_err = assess(M, ŷ, y)
    update!(M, y, ŷ, x, learn_rate)
    M.sol.iter += 1
    append!(M.inf["errors"], M.sol.state_err)
  end
end
