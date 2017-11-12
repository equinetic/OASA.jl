function assess(M::Model, y, ŷ)
  assess(M.obj, y, ŷ)
end

function assessgrad(M::Model, y, ŷ, x)
  assessgrad(M.obj, y, ŷ, x)
end

"""
L1 Distance Objective
"""
struct L1_Obj <: Objective end

function assess(O::L1_Obj, y, ŷ)
  value(L1DistLoss(), y, ŷ, AvgMode.Sum())
end

function assessgrad(O::L1_Obj, y, ŷ, x)
  deriv(L1DistLoss(), y, ŷ) * x
end

"""
L2 Distance Objective
"""
struct L2_Obj <: Objective end

function assess(O::L2_Obj, y, ŷ)
  value(L2DistLoss(), y, ŷ, AvgMode.Sum())
end

function assessgrad(O::L2_Obj, y, ŷ, x)
  deriv(L2DistLoss(), y, ŷ) * x
end


"""
Logistic Objective
"""
struct Logit_Obj <: Objective end

function assess(O::Logit_Obj, y, ŷ)
  value(LogitDistLoss(), y, ŷ, AvgMode.Sum())
end

function assessgrad(O::Logit_Obj, y, ŷ, x)
  deriv(LogitDistLoss(), y, ŷ) * x
end

#
# """
# L1 Distance Objective
# """
# struct L1 <: Objective end
#
# function loss(O::L1, ŷ, y)
#   return (1/length(y)) * sum(abs, y .- ŷ)
# end
#
# function lossgrad(O::L1, ŷ, y, x)
#   return (1/length(y)) * sign.(ŷ .- y) * x
# end

# """
# L2 Distance Objective
# """
# struct L2 <: Objective end
#
# function loss(O::L2, y, ŷ)
#   return (1/length(y)) * sum(abs2, y .- ŷ)
# end
#
# function lossgrad(O::L2, y, ŷ, x)
#   return (1/length(y)) * (ŷ .- y) * x
# end
#
# """
# Sigmoid Objective
# """
# struct Sigmoid <: Objective end
#
# function loss(O::Sigmoid, y, ŷ)
#   return 1 ./ (exp(1) .^ (y .- ŷ))
# end
#
# function lossgrad(O::Sigmoid, y, ŷ, x)
#   return Void()
# end
