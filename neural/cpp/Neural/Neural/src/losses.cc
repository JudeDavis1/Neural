#include "losses.h"

_differentiable_scalar Neural::Losses::MSE(Eigen::VectorXd labels, Eigen::MatrixXd outputs)
{
	_differentiable_scalar loss;

	loss.function = (outputs.array().colwise() - labels.array()).square().mean();
	loss.derivative = (2 * (outputs.array().colwise() - labels.array())).mean();

	return loss;
}

_differentiable_scalar Neural::Losses::CategoricalCrossentropy(Eigen::MatrixXd labels, Eigen::MatrixXd logits)
{
	_differentiable_scalar loss;

	loss.function = ((-logits).array().log()/* * labels.array()*/).mean();
	loss.derivative = ((-labels).array() / logits.array()).mean();

	return loss;
}