#pragma once

#include <vector>
#include <iostream>

#include "layers.h"
#include "optimizers.h"
#include <variant>


using namespace Neural::Layers;

namespace Neural
{
    namespace Models
    {
        class NN
        {
        private:
            AOptimizer optimizer;
            double epsilon_balancer;
            bool is_training = false;
            bool started_training = false;
            std::function<_differentiable_scalar(Eigen::MatrixXd, Eigen::MatrixXd)> loss;
        public:
            std::vector<ALayer_t> layers;

            NN(std::vector<ALayer_t>& layers);

            void Compile(AOptimizer optimizer, std::function<_differentiable_scalar(Eigen::MatrixXd, Eigen::MatrixXd)> loss);
            void _Initialize(Eigen::MatrixXd& x);
            Eigen::MatrixXd Predict(Eigen::MatrixXd& x);
            void Fit(Eigen::MatrixXd& x_train, Eigen::VectorXd& y_train, int epochs = 100);

            ~NN();
        };
    }
}