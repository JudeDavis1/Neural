#include <iostream>

#include "activations.h" // _differentiable structure

/////////////////////////////////////////////////////////////////
// NOTE:                                                       //
//                                                             //
// This list of functions will be updated in the near future...//
// Just implementing the frequent ones at the moment...        //
/////////////////////////////////////////////////////////////////

namespace Neural
{
	namespace Losses
	{
		_differentiable_scalar MSE(Eigen::VectorXd labels, Eigen::MatrixXd outputs);
		_differentiable_scalar CategoricalCrossentropy(Eigen::MatrixXd labels, Eigen::MatrixXd logits);
	}
}