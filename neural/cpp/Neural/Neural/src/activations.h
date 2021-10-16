#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <iostream>
#include "../includes/Eigen/Dense"

struct _differentiable
{
    Eigen::MatrixXd function;
    Eigen::MatrixXd derivative;
};

struct _differentiable_scalar
{
    double function;
    double derivative;
};

namespace Neural
{
    namespace Activations
    {
        _differentiable Tanh(Eigen::MatrixXd &z);
        _differentiable ReLU(Eigen::MatrixXd &z);
        _differentiable Linear(Eigen::MatrixXd &z);
        _differentiable Sigmoid(Eigen::MatrixXd &z);
        _differentiable Softmax(Eigen::MatrixXd &z);
    };
}

#endif // ACTIVATIONS_H