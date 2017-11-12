mutable struct SingleLayer <: Architecture
  wts::AbstractVecOrMat
end

mutable struct MultiLayer <: Architecture
  wts::AbstractMatrix
end


# mutable struct ANN <: Architecture
#
# end
