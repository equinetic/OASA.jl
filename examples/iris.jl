using OASA,
      RDatasets,
      UnicodePlots,
      Base.Test

# Load data
# =========
d = Matrix(dataset("datasets", "iris"))

X = Float64.(d[:, 2:4])
Y = reshape(Float64.(d[:, 1]), (1, 150))


# Define model
# ============
# Variable "X" is an Mxn matrix of independent (endogenic) features
obj = L2_Obj()                          # L2 objective: ∑ (y - ŷ)^2
alg = Linear()                          # Linear model
sol = GradientDescent()                 # Full batch gradient descent
arc = SingleLayer(zeros(1,size(X,2)))   # 1xn matrix of weights

# OASA models also contain a key-value store for miscellaneous information
# generated during training and other events
inf = Dict()

# Instantiate model
reg = Model(obj, alg, sol, arc, inf)


# Test model
# ==========
# Ensure everything is set up properly
@testset "Model Initialized" begin
  # Dimensions of inference == dimensions of Y
  @test size(infer(reg, X)) == size(Y)
  # loss is producing a floating point number greater than 0
  @test assess(reg, Y, infer(reg, X)) > 0.0
  # lossgrad dimensions == 1xn matrix
  @test size(assessgrad(reg, Y, infer(reg, X), X)) == (1, size(X,2))
end


# Train model
# ===========
train!(reg, X, Y; learn_rate = 1e-6)

# Plot errors in terminal
lineplot(reg.inf["errors"], canvas=AsciiCanvas, border=:ascii, title="Objective")

# Also note that the components of `reg` are pointers to the variables
# defined earlier. We can access this information more directly:
# lineplot(inf["errors"])
