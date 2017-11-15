module OASA

using   LossFunctions,
        ProgressMeter
        #PenaltyFunctions
        #TiledIteration

export  Model,
          Objective,
            L1_Obj,
            L2_Obj,
            Logit_Obj,
          Algorithm,
            Linear,
          Solver,
            GradientDescent,
          Architecture,
            SingleLayer,
            MultiLayer,
        infer,
        train!,
        update!,
        assess,
        assessgrad

abstract type   Objective     end
abstract type   Algorithm     end
abstract type   Solver        end
abstract type   Architecture  end

mutable struct Model
  obj::Objective
  alg::Algorithm
  sol::Solver
  arc::Architecture
  inf::Dict{Any, Any}
end

function Model(obj::Objective,
                alg::Algorithm,
                sol::Solver,
                arc::Architecture)::Model
    Model(obj, alg, sol, arc, Dict())
end

function Model(obj::Objective,
                alg::Algorithm,
                sol::Solver,
                arc::Architecture,
                inf::Dict)::Model
    Model(obj, alg, sol, arc, inf)
end

include("architectures.jl")
include("objectives.jl")
include("algorithms.jl")
include("solvers.jl")

end
