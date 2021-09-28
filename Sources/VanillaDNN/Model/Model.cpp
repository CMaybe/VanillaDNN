#ifndef VANILLA_DNN_MODEL_CPP
#define VANILLA_DNN_MODEL_CPP

#include <VanillaDNN/Model/Model.hpp>


Model::Model() {
	this->batch_size = 1;
	this->epoch = 0;
	this->nEval = 0;
	this->total = 0;
	this->accuracy = 0;
	this->error = 0;
	this->optimizer = new Optimizer();
}

Model::~Model() {
	if (this->inputLayer != nullptr) delete this->inputLayer;
	if (this->outputLayer != nullptr) delete this->outputLayer;
	for (Layer* layer : layers) {
		if (layer != nullptr) delete layer;
	}
	layers.clear();
}

void Model::setLoss(Loss _loss) {
	this->loss = std::move(std::bind(_loss, std::placeholders::_1, std::placeholders::_2));
	if (*(_loss.target<float(*)(Vector<float>&, Vector<float>&)>()) == LOSS_FUNCTION::mean_squared_error) {
		loss_diff = std::bind(DIFF_FUNCTION::mean_squared_error_diff, std::placeholders::_1, std::placeholders::_2);
	}
	else if (*(_loss.target<float(*)(Vector<float>&, Vector<float>&)>()) == LOSS_FUNCTION::root_mean_squared_error) {
		loss_diff = std::bind(DIFF_FUNCTION::root_mean_squared_error_diff, std::placeholders::_1, std::placeholders::_2);
	}
	else if (*(_loss.target<float(*)(Vector<float>&, Vector<float>&)>()) == LOSS_FUNCTION::categorical_cross_entropy) {
		loss_diff = std::bind(DIFF_FUNCTION::categorical_cross_entropy_diff, std::placeholders::_1, std::placeholders::_2);
	}
	else if (*(_loss.target<float(*)(Vector<float>&, Vector<float>&)>()) == LOSS_FUNCTION::binary_cross_entropy) {
		loss_diff = std::bind(DIFF_FUNCTION::binary_cross_entropy_diff, std::placeholders::_1, std::placeholders::_2);
	}
	else {
		std::cout << "fail\n";
	}
	return;
}

void Model::init() {
	Layer* cur = this->inputLayer;
	do {
		cur->init(this->batch_size, this->optimizer);
	} while ((cur = cur->postLayer) != nullptr);
}

void Model::feed_forward(int idx) {
	this->inputLayer->input[idx % this->batch_size] = this->input_set[idx];
	for(int i =  1;i<this->layers.size();i++) {
		layers[i]->feed_forward(idx % this->batch_size);	
	}
	this->output_set[idx % this->batch_size] = this->outputLayer->output[idx % this->batch_size];
	return;
}

void Model::back_propagation(int idx) {
	Layer* cur = this->outputLayer;
	dynamic_cast<DenseLayer*>(this->outputLayer)->dE_do[idx % this->batch_size] = 
		this->loss_diff(this->output_set[idx % this->batch_size], this->target_set[idx]);
	do{
		cur->back_propagation(idx % this->batch_size);
	}while((cur = cur->preLayer)!= nullptr);

}

void Model::update(){
	Layer* cur = this->outputLayer;
	do{
		cur->update();
	}while((cur = cur->preLayer)!= nullptr);
}

void Model::fit(int _total, int _epoch, int _batch) {
	this->total = _total;
	this->epoch = _epoch;
	this->batch_size = _batch;
	this->init();
	this->inputLayer->preLayer = nullptr;
	this->outputLayer->postLayer = nullptr;
	std::vector<std::future<void>> batch_tasks;
	for (int i = 0; i < this->epoch; i++) {
		for (int j = 0; j < this->total; j+=this->batch_size) {
			for(int k = 0; k< this->batch_size; k++) {
				batch_tasks.push_back(std::async(std::launch::async,[j,k,this](){
					this->feed_forward(j+k);
					this->back_propagation(j+k);
				}));
			}
			for(int i =0;i<batch_tasks.size();i++){
				batch_tasks[i].wait();
			}
			this->update();
		}
		std::cout << i + 1 << " epoch is done\n";
	}
	

}

void Model::evaluate(int _len,bool show) {
	this->accuracy = 0;
	float acc_sum = 0;
	float error_sum = 0;
	for (int i = 0; i < _len; i++) {
		this->target = this->target_set[i];
		this->feed_forward(i);

		this->output = this->output_set[i % this->batch_size];

		if (show) {
			std::cout << "\n\ntarget : \n";
			std::cout << this->target;
			std::cout << "output : \n";
			std::cout << this->output;
		}
		acc_sum += static_cast<float>(this->target == this->output.onehot());
		error_sum += this->loss(this->output, target_set[i]);
	}
	this->accuracy = acc_sum / _len;
	this->error = error_sum / _len;
}

float Model::getAccuracy() {
	return this->accuracy;
}

float Model::getError() {
	return this->error;
}

int Model::getDepth() {
	return this->layers.size();
}


void Model::addLayer(Layer* _layer) {
	this->layers.push_back(_layer);
	if (this->layers.size() == 1) {
		this ->inputLayer = _layer;
		this ->outputLayer = _layer;
	}
	else {
		this->outputLayer->connect(_layer);
		this->outputLayer = _layer;
	}
	return;
}


void Model::setOptimizer(Optimizer *_optimizer) {
	if(this->optimizer != nullptr) delete this->optimizer;
	this->optimizer = _optimizer;
}

void Model::setInput(std::vector<Vector<float>>& _input_set) {
	this->input_set = _input_set;
}

void Model::setTarget(std::vector<Vector<float>>& _target_set) {
	this->target_set = _target_set;
}

void Model::setLearningRate(float lr)
{
	this->optimizer->setLearningRate(lr);
}

#endif


