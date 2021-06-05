#include "DNN/Layer/Layer.hpp"
#include "DNN/Model/Model.hpp"
#include "DNN/Functions/DNNFunction.hpp"
#include "MNIST/MNIST.hpp"
#include <iostream>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
	//training set
	MNIST training_set(MNIST_DATA_DIR, "train");
	
	// Get MNIST Data
	std::vector<std::vector<char>> pre_training_images = training_set.getImages();
	std::vector<char> pre_training_labels = training_set.getLabels();
	
	
	// For learning Data
	std::vector<Vector<float>> training_images(pre_training_images.size());
	std::vector<Vector<float>> training_labels(pre_training_labels.size());

	for (int i = 0; i < pre_training_images.size(); i++) {
		training_images[i].resize(784, 0);
		for (int j = 0; j < 784; j++) {
			training_images[i][j] = pre_training_images[i][j] != 0;
		}
	}


	for (int i = 0; i < pre_training_labels.size(); i++) {
		training_labels[i].resize(10, 0);
		training_labels[i][pre_training_labels[i]] = 1;
	}

	//evaluate set
	MNIST evaluate_set(MNIST_DATA_DIR, "test");
	std::vector<std::vector<char>> pre_evaluate_images = evaluate_set.getImages();
	std::vector<char> pre_evaluate_labels = evaluate_set.getLabels();

	std::vector<Vector<float>> evaluate_images(pre_evaluate_images.size());
	std::vector<Vector<float>> evaluate_labels(pre_evaluate_labels.size());

	for (int i = 0; i < pre_evaluate_images.size(); i++) {
		evaluate_images[i].resize(784, 0);
		for (int j = 0; j < 784; j++) {
			evaluate_images[i][j] = pre_evaluate_images[i][j] != 0;
		}
	}


	for (int i = 0; i < pre_evaluate_labels.size(); i++) {
		evaluate_labels[i].resize(10, 0);
		evaluate_labels[i][pre_evaluate_labels[i]] = 1;
	}



	Model mnist(784, 10);//input : 28 x 28, output 0 ~ 9;
	
	mnist.setOutputFunction(ACTIVATION_FUNCTION::sigmoid);
	mnist.setLoss(LOSS_FUNCTION::mean_squared_error);

	mnist.addLayer(new Layer(384, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(128, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(32, ACTIVATION_FUNCTION::sigmoid));
	
	mnist.setLearningRate(0.8f);
	mnist.setInput(training_images);
	mnist.setTarget(training_labels);
	mnist.fit(50000, 10, 50); //total, epoch, batch

	mnist.setInput(evaluate_images);
	mnist.setTarget(evaluate_labels);

	std::cout << "training is done" << '\n';

	mnist.evaluate(7000);
	std::cout <<"Accuracy : "<< mnist.getAccuracy() << '\n';


	return 0;
}