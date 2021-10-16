#include "activations.h"

_differentiable Neural::Activations::Linear(Eigen::MatrixXd &z)
{
    _differentiable a;

    a.function = z;
    a.derivative = Eigen::MatrixXd::Ones(z.rows(), z.cols());

    return a;
}

_differentiable Neural::Activations::Sigmoid(Eigen::MatrixXd &z)
{
    _differentiable a;

    a.function = 1 / (1 + Eigen::exp(z.array()));
    a.derivative = a.function.array() * (1 - a.function.array());

    return a;
}

_differentiable Neural::Activations::ReLU(Eigen::MatrixXd &z)
{
    _differentiable a;

    Eigen::MatrixXd F(z.rows(), z.cols());
    Eigen::MatrixXd dF(z.rows(), z.cols());

    for (int i = 0; i < z.rows(); i++)
    {
        for (int j = 0; j < z.cols(); j++)
        {
            if (z(i, j) <= 0)
            {
                F(i, j) = 0;
                dF(i, j) = 0;
            }
            else
            {
                F(i, j) = z(i, j);
                dF(i, j) = 1;
            }
        }
    }

    a.function = F;
    a.derivative = dF;

    return a;
}

_differentiable Neural::Activations::Softmax(Eigen::MatrixXd &z)
{
    _differentiable a;

    a.function = Eigen::exp(z.array() - z.maxCoeff());
    a.function /= z.sum();
    a.derivative = a.function.array() * (1 - a.function.array());

    return a;
}

_differentiable Neural::Activations::Tanh(Eigen::MatrixXd &z)
{
    _differentiable a;

    a.function = z.array().tanh();
    a.derivative = Eigen::pow(1 / Eigen::cosh(z.array()), 2);

    return a;
}