# Overview

This package is currently in development and is not guaranteed to be stable. To install, simply clone this repository:

```julia
Pkg.clone("https://github.com/equinetic/OASA.jl.git")
```

See [TODO.md](/TODO.md) for current list of action items.

# OASA.jl

The OASA package provides the modular structure required for building and manipulating machine learning models. We organize a model into the follow adaptable components, which we refer to as the four nouns of predictive modeling:

**O**bjective - the function to minimize that describes model performance.

**A**lgorithm - defines how predictions are generated from a model's parameters and how to update those parameters during training.

**S**olver - the routine to optimize the objective function.

**A**rchitecture - a representation of how the model parameters are stored and organized.

## The OASA Paradigm

#### Creating a Model

Every OASA model must define specifications for each of the four nouns, like so:

```julia
using OASA

# Assume variable "X" is an Mxn matrix of independent (endogenic) features
obj = L2_Obj()                          # L2 objective: ∑ (y - ŷ)^2
alg = Linear()                          # Linear model
sol = GradientDescent()                 # Full batch gradient descent
arc = SingleLayer(zeros(1,size(X,2)))   # 1xn matrix of weights

# OASA models also contain a key-value store for miscellaneous information
# generated during training and other events
inf = Dict()

# Instantiate model
reg = Model(obj, alg, sol, arc, inf)
```

The empty dictionary stores miscellaneous information, such as a vector of errors generated by the objective function during gradient descent.

#### Working with a Model

With the model initialized it can now be interacted with using the OASA verbiage:

* `infer(::Model, X)` - Traverse the model's architecture as defined by the algorithm to produce predictions for the feature matrix `X`.
* `assess(::Model, y, ŷ)` - Calculate the objective function given the actual and inferred response values.
* `assessgrad(::Model, y, ŷ, x)` - Gradient of the objective function with respect to the architecture's parameter weights.
* `train!(::Model, x, y)` - Use the solver to update the architecture weights to minimize the objective function, as directed by the algorithm.
* `update!(::Model, y, ŷ, x)` - Update architecture parameter weights as directed by the algorithm.

With our dialect outlined we may now train the model. A good idea is to first test for common errors.

```julia
# Assume variable "Y" contains the dependent (exogenic) response
using Base.Test

# Ensure everything is set up properly
@testset "Model Initialized" begin
  # Dimensions of inference == dimensions of Y
  @test size(infer(reg, X)) == size(Y)
  # Objective function is producing a floating point number greater than 0
  @test infer(reg, y, infer(reg, X)) > 0.0
  # Gradient dimensions == 1xn matrix
  @test size(assessgrad(reg, y, infer(reg, X), X)) == (1, size(X,2))
end
```
```
Test Summary:         | Pass  Total
Model Initialized     |    3      3
```

If everything looks good, it's time to train:

```julia
using UnicodePlots

# Train model
train!(reg, X, Y; learn_rate=1e-7, max_iter=2500)

# Plot errors in terminal
lineplot(reg.inf["errors"], canvas=AsciiCanvas, border=:ascii, title="Objective")

# Also note that the components of `reg` are pointers to the variables
# defined earlier. We can access this information more directly:
# lineplot(inf["errors"])
```

```
                Objective
+----------------------------------------+
|                                        |
|.                                       |
||                                       |
|\                                       |
|l                                       |
||                                       |
| |                                      |
| \                                      |
| ].                                     |
|  .                                     |
|  ].                                    |
|   \                                    |
|   "\.                                  |
|     \..                                |
|       ""-----\_________________________|
+----------------------------------------+
0                                     1000
```
