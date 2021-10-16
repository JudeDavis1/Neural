#include "layers.h"
#include "neural.h"

#include <random>

Neural::Layers::Dense::Dense(int n_nodes, std::function<_differentiable(Eigen::MatrixXd&)> activation)
{
    this->n_nodes = n_nodes;
    this->activation = activation;
}

void Neural::Layers::Dense::_Build(Eigen::MatrixXd& x)
{
    this->x = x;
    this->biases = Eigen::VectorXd::Zero(this->n_nodes);
    this->kernel = Eigen::MatrixXd::Zero(x.cols(), this->n_nodes);

    std::random_device device;
    std::default_random_engine gen(device());   // the engine which generates random values

    for (int i = 0; i < this->kernel.rows(); i++)
        for (int j = 0; j < this->kernel.cols(); j++)
            this->kernel(i, j) = std::uniform_real_distribution<double>(0, .1)(gen);
}

_differentiable Neural::Layers::Dense::Forward()
{
    Eigen::MatrixXd feature_kernel = (this->x * this->kernel).array().rowwise() + this->biases.array().transpose();
    
    this->outputs = this->activation(feature_kernel);
    
    return this->outputs;
}

Neural::Layers::Dense::~Dense() {}

Neural::Layers::experimental::Dropout::Dropout(float rate)
{
    this->rate = 1 - rate;
}

void Neural::Layers::experimental::Dropout::_Build(Eigen::MatrixXd& x)
{
    this->x = x;
    this->kernel = Eigen::MatrixXd::Zero(x.rows(), x.cols());

    std::random_device device;
    std::default_random_engine gen(device());

    for (int i = 0; i < this->kernel.rows(); i++)
        for (int j = 0; j < this->kernel.cols(); j++)
            this->kernel(i, j) = std::binomial_distribution<>(1, this->rate)(gen) / this->rate;
}

_differentiable Neural::Layers::experimental::Dropout::Forward()
{
    _differentiable a;

    a.function = (this->x * this->kernel.transpose()).array();
    a.derivative = this->kernel;

    this->outputs = a;

    return this->outputs;
}