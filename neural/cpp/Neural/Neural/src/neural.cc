#include "neural.h"
#include <random>

int Neural::Random::RandN(int max)
{
    time_t t;
    srand( (unsigned) time(&t) );

    return 1 + (rand() % max);
}

double Neural::Random::RandN()
{
    time_t t;
    srand( (unsigned) time(&t) );

    return rand();
}

float Neural::WorkingAccuracy(Eigen::MatrixXd& prediction, Eigen::VectorXd& y_)
{
    Eigen::MatrixXd accuracy(prediction.rows(), 1);
    for (int i = 0; i < prediction.rows(); i++)
    {
        int pred = round( prediction(i) );
        int y_true = round( y_(i) );

        accuracy(i) = 0;
        if (pred == y_true)
            accuracy(i) = 1;
    }

    return static_cast<float> (accuracy.mean());
}