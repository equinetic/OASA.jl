# TODO

- [x] Remove algorithm logic from gradient descent solver

- [x] Logistic regression

- [ ] Change algo/arc names. "Linear" for architecture and "Regression" for algorithm sounds like the same thing, I think architecture needs more technical terminology while the algorithm should adhere to norms.

- [ ] Add PenaltyFunctions

- [ ] Add arguments for gradient descent error log

- [ ] Precompile

- [ ] Regression -> "Linear". Make it work with multiple layers. Consider incorporating tiled iteration.

- [ ] Enable use of link function, i.e. for GLM
  - May need to rethink Objective representation for this. Maybe it should be its own struct with field (ObjType, LinkFun, Regularizer). Here ObjType would be used for dispatching infer and update, LinkFun would be a transformer post forward pass (not sure how this works for backprop yet), and finally Regularizer would incorporate penalty functions and other methods (e.g. killing nodes).
