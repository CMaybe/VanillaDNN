#ifndef VANILLA_DNN_MODEL_CPP
#define VANILLA_DNN_MODEL_CPP

#include <VanillaDNN/Model/Model.hpp>


Model::Model(int _nInput, int _nOutput) {
	this->depth = 0;
	this->nInput = _nInput;
	this->nOutput = _nOutput;
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
	this->output_set.resize(this->batch_size);
	this->inputLayer->preLayer = nullptr;
	this->inputLayer->inputNeuron.resize(this->batch_size);
	this->inputLayer->outputNeuron.resize(this->batch_size);
	Layer* cur = this->outputLayer;
	do {
		cur->weight.resize(cur->getNueronCnt(), cur->preLayer->getNueronCnt());
		cur->weight.setRandom();
		cur->bias.resize(cur->getNueronCnt());
		cur->inputNeuron.resize(this->batch_size);
		cur->outputNeuron.resize(this->batch_size);
		cur->dE_dw.resize(this->batch_size);
		cur->dE_do.resize(this->batch_size);
		cur->do_dz.resize(this->batch_size);
		cur->dz_dw.resize(this->batch_size);
		cur->dE_db.resize(this->batch_size);
		cur->dE_dz.resize(this->batch_size);
		cur->dz_db.resize(this->batch_size);
		for(int i = 0; i < this->batch_size; i++){
			cur->dE_dw[i].resize(cur->getNueronCnt(), cur->preLayer->getNueronCnt(), 0);
			cur->dE_do[i].resize(cur->getNueronCnt(), 0);
			cur->do_dz[i].resize(cur->getNueronCnt(), 0);
			cur->dz_dw[i].resize(cur->preLayer->getNueronCnt(),0);
			cur->dE_db[i].resize(cur->getNueronCnt(), 0);
			cur->dE_dz[i].resize(cur->getNueronCnt(), 0);
			cur->dz_db[i].resize(cur->getNueronCnt(), 0);
		}
		
		batch_dE_dw.insert(batch_dE_dw.begin(), Matrix<float>(cur->getNueronCnt(), cur->preLayer->getNueronCnt(), 0));
		batch_dE_db.insert(batch_dE_db.begin(), Vector<float>(cur->getNueronCnt(), 0));
		
		cur = cur->preLayer;
	} while (cur->preLayer != nullptr);
}

void Model::feed_forward(int idx) {
	this->inputLayer->inputNeuron[idx % this->batch_size] = this->input_set[idx];
	this->inputLayer->outputNeuron[idx % this->batch_size] = this->input_set[idx];
	for (Layer* layer : this->layers) {
		layer->feed_forward(idx % this->batch_size);	
	}
	this->output_set[idx % this->batch_size] = this->outputLayer->outputNeuron[idx % this->batch_size];
	return;
}

void Model::back_propagation(int idx) {
	Layer* cur = nullptr;
	int _depth = this->depth;
	//output Layer
	cur = this->outputLayer;
	this->outputLayer->dE_do[idx % this->batch_size] = this->loss_diff(this->output_set[idx % this->batch_size], this->target_set[idx]);
	do{
		cur->back_propagation(idx % this->batch_size);
	}while((cur = cur->preLayer)!= nullptr);

}

void Model::update(){
	
	for(int i = 0; i < this->batch_size; i++){
		for(int j = 1; j < layers.size(); j++){
			if(layers[j]->preLayer == nullptr) continue;
			this->batch_dE_dw[j] += layers[j]->dE_dw[i];
			this->batch_dE_db[j] += layers[j]->dE_db[i];
		}
	}
	
	for(int i =  1;i < layers.size(); i++){
		layers[i]->update(this->optimizer->getWeightGradient(this->batch_dE_dw[i], i) / this->batch_size, this->batch_dE_db[i]  / this->batch_size);
	}
	
	
	
	batch_dE_dw.clear();
	batch_dE_db.clear();
	
	Layer *cur = this->outputLayer;
	do {
		
		batch_dE_dw.insert(batch_dE_dw.begin(), Matrix<float>(cur->getNueronCnt(), cur->preLayer->getNueronCnt(), 0));
		batch_dE_db.insert(batch_dE_db.begin(), Vector<float>(cur->getNueronCnt(), 0));
		
		cur = cur->preLayer;
	} while (cur->preLayer != nullptr);
}

void Model::fit(int _total, int _epoch, int _batch) {
	this->total = _total;
	this->epoch = _epoch;
	this->batch_size = _batch;
	this->init();
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
	this->layers[0]->preLayer = inputLayer;
	this->inputLayer->preLayer = nullptr;
	this->outputLayer->preLayer = this->layers[this->depth - 1];
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
	return this->depth;
}


void Model::addLayer(Layer* _layer) {
	this->layers.push_back(_layer);
	this->depth = layers.size();
	if (this->layers.size() == 1) {
		this ->inputLayer = this->layers[0];
		this ->outputLayer = this->layers[0];
	}
	else {
		this->outputLayer->connect(_layer);
		this->outputLayer = _layer;
	}
	return;
}

void Model::addLayers(std::vector<Layer*>& _layers) {
	this->depth += _layers.size();
	for (Layer* layer : _layers) {
		this->layers.push_back(layer);
		if (layers.size() != 1) {
			layers[layers.size() - 1]->preLayer = layers[layers.size() - 2];
		}
	}
	if (layers.size() == _layers.size()) {
		layers[0]->preLayer = inputLayer;
	}
	else {
		outputLayer->preLayer = layers[depth - 1];
		layers[depth - 1]->preLayer = layers[depth - 2];
	}
	outputLayer->preLayer = layers[depth - 1];
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


