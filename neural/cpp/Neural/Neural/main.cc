//
//  main.cpp
//  Neural
//
//  Created by Jude Davis on 26/03/2021.
//

#include <cmath>
#include <random>
#include <variant>
#include <iostream>


// local libraries
#include "includes/Eigen/Dense"
#include "src/neural.h"

int main(int argc, const char* argv[])
{
    Eigen::MatrixXd x = (
        Eigen::MatrixXd(5, 1) << 1, 2, 3, 4, 5
    ).finished();
    Eigen::VectorXd y = (
        Eigen::VectorXd(5) << 0, 1, 1, 0, 0
    ).finished();

    /*
    x = (x.array() - x.array().minCoeff()) / (x.maxCoeff() - x.minCoeff());
    y = (y.array() - y.array().minCoeff()) / (y.maxCoeff() - y.minCoeff());
    */

    auto layers = std::vector<ALayer_t>{
        Neural::Layers::Dense(10, Neural::Activations::Sigmoid),
        Neural::Layers::Dense(y.rows(), Neural::Activations::Sigmoid)
    };

    Neural::Models::NN model(layers);

    model.Compile(Neural::Optimizers::SGD(0.001), Neural::Losses::MSE);
    model.Fit(x, y, 10000);
    Eigen::MatrixXd prediction = model.Predict(x).array().rowwise().mean();
    std::cout << Neural::WorkingAccuracy(prediction, y) * 100 << "% Accuracy" << std::endl;

    std::cin.get();
}
