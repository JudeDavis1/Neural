#pragma once

#include <iostream>

// local libraries
#include "activations.h"
#include <variant>

class AbstractLayer
{
public:
    Eigen::MatrixXd kernel;
    _differentiable outputs;

    virtual void _Build(Eigen::MatrixXd& x)=0;
    virtual _differentiable Forward()=0;
};


namespace Neural
{
    namespace Layers
    {
        class Dense: public AbstractLayer
        {
        public:
            // variables
            Eigen::MatrixXd x;
            Eigen::MatrixXd kernel;
            Eigen::VectorXd biases;
            _differentiable outputs;

            int n_nodes;
            std::function<_differentiable(Eigen::MatrixXd&)> activation;

            // methods
            Dense(int n_nodes, std::function<_differentiable(Eigen::MatrixXd&)> activation);

            void _Build(Eigen::MatrixXd& x);
            _differentiable Forward();
            
            ~Dense();
        };

        namespace experimental
        {
            class Dropout: public AbstractLayer
            {
            public:
                float rate;
                Eigen::MatrixXd x;
                Eigen::MatrixXd kernel;
                _differentiable outputs;

                Dropout(float rate = 0.4);
                void _Build(Eigen::MatrixXd& x);
                _differentiable Forward();
            };
        }
    }
}
using namespace Neural::Layers;

using ALayer_t = std::variant<Dense, experimental::Dropout>;