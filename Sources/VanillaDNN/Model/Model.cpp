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
	this->inputLayer = nullptr;
	this->outputLayer = nullptr;
	this->optimizer = std::make_unique<Optimizer>();
}

Model::~Model() {
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
	this->output_set.resize(this->batch_size);
	auto cur = this->inputLayer;
	do {
		cur->init(this->batch_size, this->optimizer);
	} while ((cur = cur->getPostLayer()).use_count() > 0);

}


void Model::feed_forward(int idx) {
	this->inputLayer->setInput(this->input_set[idx], idx % this->batch_size);
	auto cur = this->inputLayer;
	do {
		cur->feed_forward(idx % this->batch_size);
		cur = cur->getPostLayer();
	} while (cur.use_count() > 0);
	this->output_set[idx % this->batch_size] = this->outputLayer->getOutput(idx % this->batch_size);
	return;
}

void Model::predict(int idx){
	this->inputLayer->setInput(this->input_set[idx], 0);
	auto cur = this->inputLayer;
	do {
		cur->predict();	
	} while ((cur = cur->getPostLayer()).use_count() > 0);
	this->output = this->outputLayer->getOutput(0);
	return;
}


void Model::back_propagation(int idx) {
	auto cur = this->outputLayer;
	this->outputLayer->setError(
		this->loss_diff(this->output_set[idx % this->batch_size], this->target_set[idx]), idx % this->batch_size);
	do{
		cur->back_propagation(idx % this->batch_size);
	}while((cur = cur->getPreLayer()).use_count() > 0);
	return;
}

void Model::update(){
	auto cur = this->outputLayer;
	do{
		cur->update();
	}while((cur = cur->getPreLayer()).use_count() > 0);

	return;
}

void Model::fit(int _total, int _epoch, int _batch) {
	this->total = _total;
	this->epoch = _epoch;
	this->batch_size = _batch;
	this->init();
	std::vector<std::future<void>> batch_tasks;
	for (int i = 0; i < this->epoch; i++) {
		for (int j = 0; j < this->total; j+=this->batch_size) {
			for(int k = 0; k< this->batch_size && (j+k)< this->total; k++) {
				batch_tasks.emplace_back(std::async(std::launch::async,[j,k,this](){
					this->feed_forward(j+k);
					this->back_propagation(j+k);
				}));
			}
			for(int i =0;i<batch_tasks.size();i++){
				batch_tasks[i].wait();
			}
			this->update();
			batch_tasks.clear();
			batch_tasks.reserve(this->batch_size);
		}
		std::cout << i + 1 << " epoch is done\n";
	}
	
	return;
}

void Model::evaluate(int _len,bool show) {
	this->accuracy = 0;
	float acc_sum = 0;
	float error_sum = 0;
	for (int i = 0; i < _len; i++) {
		this->target = this->target_set[i];
		this->predict(i);
		if (show) {
			std::cout << "\n\ntarget : \n";
			std::cout << this->target;
			std::cout << "output : \n";
			std::cout << this->output;
		}
		acc_sum += static_cast<float>(this->target == this->output.onehot());
		error_sum += this->loss(this->output, target);
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



void Model::addLayer(std::shared_ptr<Layer> layer) {
	if (this->outputLayer == nullptr) {
		this->inputLayer = layer;
		this->outputLayer = layer;
	}
	else {
		this->outputLayer->connect(outputLayer ,layer);
		this->outputLayer = layer;
	}
	return;
}


void Model::setOptimizer(Optimizer *_optimizer) {
	this->optimizer = std::unique_ptr<Optimizer>(_optimizer);
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


