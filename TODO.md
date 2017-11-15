# TODO

#### Package Level
- [ ] Enable Precompile
- [ ] Once enough of a stable baseline is built split the nouns into subdirectories.

#### Recipe Capabilities
- [ ] Ridge Regression
- [ ] Lasso Regression
- [ ] Decision Tree
- [ ] MultiLayer Perceptron
- [ ] KMeans Clustering
- [ ] Hierarchical Clustering
- [ ] Random Forest

#### Objectives
- [x] Logistic
- [ ] Enable use of link function, i.e. for GLM
  - May need to rethink Objective representation for this. Maybe it should be its own struct with field (ObjType, LinkFun, Regularizer). Here ObjType would be used for dispatching infer and update, LinkFun would be a transformer post forward pass (not sure how this works for backprop yet), and finally Regularizer would incorporate penalty functions and other similar methods (e.g. killing nodes).
- [ ] Incorporate PenaltyFunctions

#### Algorithms
- [x] Regression -> "Linear"

#### Solvers
- [x] Remove algorithm logic from gradient descent solver
- [ ] Make GradientDescent converge based on error delta
- [ ] Add arguments for gradient descent error log
- [ ] Add functioning batch size, build with paralellization in mind

#### Architectures
- [ ] MultiLayer. Consider incorporating tiled iteration.
