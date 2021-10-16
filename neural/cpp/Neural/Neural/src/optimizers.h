#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <iostream>

#include "layers.h"
#include "activations.h"  // differentiable structures


namespace Neural
{
	namespace Optimizers
	{
		class SGD
		{
		private:
			float lr;
			float rate;
		public:
			bool is_linear;

			Eigen::MatrixXd x;
			Eigen::MatrixXd y;
			Eigen::MatrixXd output;
			_differentiable activation;
			_differentiable_scalar loss;

			SGD(float lr = 0.01, float epsilon_balance_rate = 0.01);

			void _Propagate(ALayer_t& layer);

			~SGD();
		};
	}
}

typedef Neural::Optimizers::SGD AOptimizer;

#endif
