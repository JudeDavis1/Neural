#pragma once

#include <iostream>
#include "../includes/Eigen/Dense"

// Local libraries
#include "models.h"
#include "layers.h"
#include "losses.h"
#include "optimizers.h"
#include "activations.h"

#include <variant>


namespace Neural
{
	namespace Random
	{
		int RandN(int max);
		double RandN();
	}

	float WorkingAccuracy(Eigen::MatrixXd& prediction, Eigen::VectorXd& y_true);
}
