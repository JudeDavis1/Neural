#include "neural.h"

Neural::Optimizers::SGD::SGD(float lr, float epsilon_balance_rate)
{
	this->lr = lr;
	this->rate = epsilon_balance_rate;
}

void Neural::Optimizers::SGD::_Propagate(ALayer_t& layer)
{
	Eigen::MatrixXd dw;
	Eigen::MatrixXd db;
	size_t index = layer.index();

	dw = (this->x(Neural::Random::RandN(this->x.rows() - 1)) * this->activation.derivative * this->loss.derivative);
	db = (this->activation.derivative * this->loss.derivative);

	switch (index)
	{
	case 0:
		for (int i = 0; i < std::get<0>(layer).kernel.rows(); i++)
			std::get<0>(layer).kernel(i) = std::get<0>(layer).kernel.array()(i) - this->lr * (dw.array()(i) + this->rate);

		for (int i = 0; i < std::get<0>(layer).biases.rows(); i++)
			std::get<0>(layer).biases(i) = std::get<0>(layer).biases.array()(i) - this->lr * db.array()(i);
		break;
	case 1:
		for (int i = 0; i < std::get<0>(layer).kernel.rows(); i++)
			std::get<1>(layer).kernel(i) = std::get<0>(layer).kernel.array()(i) - this->lr * (dw.array()(i) + this->rate);
		break;
	}
}

Neural::Optimizers::SGD::~SGD() {}