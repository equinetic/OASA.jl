using OASA
using Base.Test

# Synthesize random data
X = hcat(ones(100, 1), rand(100, 3))
truth_theta = [.5 1.2 3.3 5.0]
Y = truth_theta * X'

# Run some basic tests
function TestA()
    yh = infer(myModel, X)
    @test size(infer(myModel, X)) == size(Y)
    @test assess(myModel, Y, yh) > 0.0
    @test size(assessgrad(myModel, Y, yh, X)) == (1, size(X,2))
end

# Ensure that all of the objective functions are operational
@testset "Objective Functions" begin
    @testset "L1 Regression" begin
        myModel = Model(L1_Obj(),
                    Linear(),
                    GradientDescent(),
                    SingleLayer(zeros(1, size(X,2))),
                    Dict())
        TestA()
    end
    @testset "L2 Regression" begin
        myModel.obj = L2_Obj()
        TestA()
    end
    @testset "Logistic Regression" begin
        myModel.obj = Logit_Obj()
        TestA()
    end
end
