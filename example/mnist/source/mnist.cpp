#include <VanillaDNN/DNN/Layers/Layer.hpp>
#include <VanillaDNN/DNN/Model/Model.hpp>
#include <VanillaDNN/DNN/Functions/DNNFunction.hpp>
#include <VanillaDNN/MNIST/MNIST.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
	//training set
	MNIST training_set(MNIST_DATA_LOCATION, "train");
	std::vector<Vector<float>> training_images(training_set.getImages());
	std::vector<Vector<float>> training_labels(training_set.getLabels());

	for (int i = 0; i < training_images.size(); i++) {
		training_images[i] /= 255.0f;
	}

	//evaluate set
	MNIST evaluate_set(MNIST_DATA_LOCATION, "test");
	std::vector<Vector<float>> evaluate_images(evaluate_set.getImages());
	std::vector<Vector<float>> evaluate_labels(evaluate_set.getLabels());

	for (int i = 0; i < evaluate_images.size(); i++) {
		evaluate_images[i] /= 255.0f;
	}

	std::cout<< "Data processing is done!\n";
	
	Model mnist(784, 10);//input : 28 x 28, output 0 ~ 9;
	
	mnist.setOutputFunction(ACTIVATION_FUNCTION::sigmoid);
	mnist.setLoss(LOSS_FUNCTION::mean_squared_error);
	mnist.addLayer(new Layer(384, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(128, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(32, ACTIVATION_FUNCTION::sigmoid));
	
	mnist.setLearningRate(0.8f);
	mnist.setInput(training_images);
	mnist.setTarget(training_labels);
	mnist.fit(5000, 10, 10); //total, epoch, batch
	mnist.setInput(evaluate_images);
	mnist.setTarget(evaluate_labels);

	std::cout << "training is done!" << '\n';

	mnist.evaluate(7000);
	std::cout <<"Accuracy : "<< mnist.getAccuracy() << '\n';


	return 0;
}