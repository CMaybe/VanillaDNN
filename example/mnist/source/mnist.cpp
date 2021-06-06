#include <VanillaDNN/DNN/Layers/Layer.hpp>
#include <VanillaDNN/DNN/Model/Model.hpp>
#include <VanillaDNN/DNN/Functions/Functions.hpp>
#include <VanillaDNN/DNN/Functions/Optimizer.hpp>
#include <VanillaDNN/MNIST/MNIST.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
	//training set
	MNIST training_set(MNIST_DATA_LOCATION, "train", ONE_HOT);
	std::vector<Vector<float>> training_images(training_set.getImages());
	std::vector<Vector<float>> training_labels(training_set.getLabels());

	//evaluate set
	MNIST evaluate_set(MNIST_DATA_LOCATION, "test", ONE_HOT);
	std::vector<Vector<float>> evaluate_images(evaluate_set.getImages());
	std::vector<Vector<float>> evaluate_labels(evaluate_set.getLabels());


	std::cout<< "Data processing is done!\n";
	
	Model mnist(784, 10);//input : 28 x 28, output 0 ~ 9;
	
	mnist.setOutputFunction(ACTIVATION_FUNCTION::sigmoid);
	mnist.setLoss(LOSS_FUNCTION::mean_squared_error);
	mnist.addLayer(new Layer(384, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(128, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(32, ACTIVATION_FUNCTION::sigmoid));
	
	// mnist.setOptimizer(new Momentum(0.1,0.9,mnist.getDepth()));
	// mnist.setOptimizer(new Adagrad(0.01f,1e-6,mnist.getDepth()));
	mnist.setOptimizer(new RMSProp(0.01f, 0.9, 1e-6,mnist.getDepth())); //lr, _rho, _epsilon, _depth
	mnist.setLearningRate(0.8f);
	mnist.setInput(training_images);
	mnist.setTarget(training_labels);
	mnist.fit(5000, 5, 10); //total, epoch, batch
	
	mnist.setInput(evaluate_images);
	mnist.setTarget(evaluate_labels);

	std::cout << "training is done!" << '\n';

	mnist.evaluate(7000);
	std::cout <<"Accuracy : "<< mnist.getAccuracy() << '\n';


	return 0;
}