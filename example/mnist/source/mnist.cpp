#include <VanillaDNN/Layers/DenseLayer.hpp>
#include <VanillaDNN/Model/Model.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>
#include <VanillaDNN/MNIST/MNIST.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
	//training set
	MNIST training_set(MNIST_DATA_LOCATION, "train");
	std::vector<Vector<float>> training_images(training_set.getImages());
	std::vector<Vector<float>> training_labels(training_set.getLabels());
	
	
	for(int i = 0;i<training_images.size();i++){
		training_images[i] /= 255.0f;
	}
	
	
	//evaluate set
	MNIST evaluate_set(MNIST_DATA_LOCATION, "test");
	std::vector<Vector<float>> evaluate_images(evaluate_set.getImages());
	std::vector<Vector<float>> evaluate_labels(evaluate_set.getLabels());

	
	for(int i = 0;i<evaluate_images.size();i++){
		evaluate_images[i] /= 255.0f;
	}

	std::cout<< "mnist loaded!\n";

	Model mnist;//input : 28 x 28, output 0 ~ 9;
	
	
	mnist.setLoss(LOSS_FUNCTION::categorical_cross_entropy);
	mnist.addLayer(new DenseLayer(784, "sigmoid"));
	mnist.addLayer(new DenseLayer(256, "sigmoid"));
	mnist.addLayer(new DenseLayer(128, "sigmoid"));
	mnist.addLayer(new DenseLayer(32, "sigmoid"));
	mnist.addLayer(new DenseLayer(10, "soft_max"));
	
	// mnist.setOptimizer(new Momentum(0.1f,0.9));
	// mnist.setOptimizer(new Adagrad(0.01f,1e-6));
	// mnist.setOptimizer(new RMSProp(0.01f, 0.9, 1e-8)); //lr, _rho, _epsilon, _depth
	mnist.setOptimizer(new Adam(0.01f, 0.9f, 0.999f, 1e-8)); //lr, _rho, _epsilon, _depth
	
	mnist.setInput(training_images);
	mnist.setTarget(training_labels);
	mnist.fit(2000, 10, 32); //total, epoch, batch
	
	mnist.setInput(evaluate_images);
	mnist.setTarget(evaluate_labels);

	std::cout << "training is done!" << '\n';

	mnist.evaluate(10000);
	std::cout <<"Accuracy : "<< mnist.getAccuracy() << '\n';


	return 0;
}