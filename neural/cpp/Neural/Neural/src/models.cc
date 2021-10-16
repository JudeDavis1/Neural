#include "neural.h"


Neural::Models::NN::NN(std::vector<ALayer_t>& layers)
{
	this->layers = layers;
}

Eigen::MatrixXd Neural::Models::NN::Predict(Eigen::MatrixXd& x)
{
	if (!this->started_training)
		this->_Initialize(x);
	else
	{
		for (int i = 1; i < this->layers.size(); i++)
		{
			_differentiable prev_layer_output;
			size_t index = this->layers[i].index();
			size_t last_index = this->layers[i - 1].index();

			/*
				Each case block may do something different for the layer. 
				
				e.g: Convolutional layer may optimize weights differently

				This is because std::get doesn't allow for variables to
				retrieve indexes, it needs constant expressions (constexpr).
				
				This makes the case checking a little bit messy and long but 
				you can't always reuse code. For now it will stay like this, but 
				I will try and check for better options in the future.
			*/

			// Get the previous layer output
			switch (last_index)
			{
			case 0:
				prev_layer_output = std::get<0>(this->layers[i - 1]).Forward();
				break;
			case 1:
				if (this->is_training)
					prev_layer_output = std::get<1>(this->layers[i - 1]).Forward();
				break;
			}

			// Get current prediction based on previous layer output
			switch (index)
			{
			case 0:
				std::get<0>(this->layers[i]).x = prev_layer_output.function;
				break;
			case 1:
				if (this->is_training)
					std::get<1>(this->layers[i]).x = prev_layer_output.function;

				break;
			}
		}
	}
	
	/* Final Prediction (switch-case block still required) */
	size_t index = this->layers[this->layers.size() - 1].index();
	switch (index)
	{
	case 0:
		return std::get<0>(this->layers[this->layers.size() - 1]).Forward().function;
		break;
	case 1:
		break;
	}
}

/* Assign the model an optimizer and a loss function */
void Neural::Models::NN::Compile(AOptimizer optimizer, std::function<_differentiable_scalar(Eigen::MatrixXd, Eigen::MatrixXd)> loss)
{
	this->loss = loss;
	this->optimizer = optimizer;
}

/* Trains the model such that all kernel are accurately set */
void Neural::Models::NN::Fit(Eigen::MatrixXd& x_train, Eigen::VectorXd& y_train, int epochs)
{
	auto outputs = this->Predict(x_train);
	
	this->is_training = true;
	this->started_training = true;

	for (int i = 0; i < epochs; i++)
	{
		outputs = this->Predict(x_train);
		_differentiable_scalar loss = this->loss(y_train, outputs);
		std::cout << loss.function << std::endl;
		// std::cout << Neural::WorkingAccuracy(outputs, y_train) * 100 << "% Accuracy" << std::endl;

		for (int j = 0; j < this->layers.size(); j++)
		{
			size_t last_index;
			size_t index = this->layers[j].index();
			
			// Allow for the first layer in the network
			if (j == 0)
				last_index = 0;
			else
				last_index = this->layers[j - 1].index();
			_differentiable layer_output;

			// Get the layer output for each case of a layer
			switch (index)
			{
			case 0:
				layer_output = std::get<0>(this->layers[j]).Forward();
				break;
			case 1:
				layer_output = std::get<1>(this->layers[j]).Forward();
				break;
			}

			this->optimizer.output = layer_output.function;
			this->optimizer.loss = this->loss(y_train, layer_output.function);
			this->optimizer.activation = layer_output;

			if (j == 0)
				this->optimizer.x = x_train;
			else
				// Set the input value for the optimizer for each case of a layer
				switch (last_index)
				{
				case 0:
					this->optimizer.x = std::get<0>(this->layers[j - 1]).outputs.function;
					break;
				case 1:
					this->optimizer.x = std::get<1>(this->layers[j - 1]).outputs.function;
					break;
				}

			switch (index)
			{
			case 0:
				this->optimizer._Propagate(this->layers[j]);
			case 1: // Ignore the Dropout layer for backprop
				break;
			}
		}
	}

	this->is_training = false;
}

void Neural::Models::NN::_Initialize(Eigen::MatrixXd& x)
{
	// For the first layer
	switch (this->layers[0].index())
	{
	case 0:
		std::get<0>(this->layers[0]).x = x;
		std::get<0>(this->layers[0])._Build(x);
		break;
	case 1:
		std::get<1>(this->layers[0]).x = x;
		std::get<1>(this->layers[0])._Build(x);
		break;
	}

	for (int i = 1; i < this->layers.size(); i++)
	{
		_differentiable prev_layer_output;
		size_t index = this->layers[i].index();  // Variant index of the current layer
		size_t last_index = this->layers[i - 1].index();  // Variant index of the last layer (for accessing the output)

		/*
			When Initializing, the following code will 
			connect the previous nodes to the output nodes 
			and will match the input and outputs for each 
			case of layer. 
			
			E.g: the previous Dense layer output is fed into 
			the Dropout layer.
		*/

		// Set the previous layer output to the output of the last layer
		switch (last_index)
		{
		case 0:
			prev_layer_output = std::get<0>(this->layers[i - 1]).Forward();
			break;
		case 1:
			prev_layer_output = std::get<1>(this->layers[i - 1]).Forward();
			break;
		}

		// Feed that last layer output into the build function of the current layer
		switch (index)
		{
		case 0:
			std::get<0>(this->layers[i])._Build(prev_layer_output.function);
			break;
		case 1:
			std::get<1>(this->layers[i])._Build(prev_layer_output.function);
			break;
		}
	}
}

Neural::Models::NN::~NN() {}
