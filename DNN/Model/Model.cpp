#ifndef VANILLA_DNN_MODEL_CPP
#define VANILLA_DNN_MODEL_CPP

#include "Model.hpp"


Model::Model(int _nInput, int _nOutput) {
	this->depth = 0;
	this->learning_rate = 1.0;
	this->nInput = _nInput;
	this->nOutput = _nOutput;
	this->inputLayer = new Layer(_nInput);
	this->outputLayer = new Layer(_nOutput);
	this->batch = 1;
	this->epoch = 0;
	this->nEval = 0;
	this->total = 0;
	this->accuracy = 0;
	this->error = 0;
}

Model::~Model() {
	if (this->inputLayer != nullptr) delete this->inputLayer;
	if (this->outputLayer != nullptr) delete this->outputLayer;
	for (Layer* layer : Layers) {
		if (layer != nullptr) delete layer;
	}
	Layers.clear();
}

void Model::setLoss(Loss _loss) {
	this->loss = std::move(std::bind(_loss, std::placeholders::_1, std::placeholders::_2));
	if (*(_loss.target<float(*)(Vector<float>, Vector<float>)>()) == LOSS_FUNCTION::mean_squared_error) {
		loss_diff = std::bind(DIFF_FUNCTION::mean_squared_error_diff, std::placeholders::_1, std::placeholders::_2);
	}
	else if (*(_loss.target<float(*)(Vector<float>, Vector<float>)>()) == LOSS_FUNCTION::root_mean_squared_error) {
		loss_diff = std::bind(DIFF_FUNCTION::root_mean_squared_error_diff, std::placeholders::_1, std::placeholders::_2);
	}
	else if (*(_loss.target<float(*)(Vector<float>, Vector<float>)>()) == LOSS_FUNCTION::cross_entropy_error) {
		loss_diff = std::bind(DIFF_FUNCTION::cross_entropy_error_diff, std::placeholders::_1, std::placeholders::_2);
	}
	else if (*(_loss.target<float(*)(Vector<float>, Vector<float>)>()) == LOSS_FUNCTION::binary_cross_entropy) {
		loss_diff = std::bind(DIFF_FUNCTION::binary_cross_entropy_diff, std::placeholders::_1, std::placeholders::_2);
	}
	else {
		std::cout << "fail\n";
	}
	return;
}

void Model::init() {
	this->Layers[0]->preLayer = inputLayer;
	this->inputLayer->preLayer = nullptr;
	this->outputLayer->preLayer = this->Layers[this->depth - 1];
	Layer* cur = this->outputLayer;
	do {
		cur->weight.resize(cur->getNueronCnt(), cur->preLayer->getNueronCnt());
		cur->weight.setRandom();
		cur->dE_dw.resize(cur->getNueronCnt(), cur->preLayer->getNueronCnt(), 0);
		cur->dz_dw.resize(cur->preLayer->getNueronCnt());
		cur->dE_db.resize(cur->getNueronCnt(), 0);
		cur = cur->preLayer;
	} while (cur->preLayer != nullptr);
}

void Model::feed_forward(int idx) {

	this->inputLayer->inputNeuron = this->input_set[idx];
	this->inputLayer->outputNeuron = this->inputLayer->inputNeuron;

	for (Layer* layer : Layers) {
		layer->inputNeuron = (layer->weight * layer->preLayer->outputNeuron) + layer->bias;
		layer->outputNeuron = layer->activation(layer->inputNeuron);
	}
	Layer* cur = this->outputLayer;
	cur->inputNeuron = (cur->weight * cur->preLayer->outputNeuron) + cur->bias;
	cur->outputNeuron = cur->activation(cur->inputNeuron);
	this->output = cur->outputNeuron;
	return;
}

void Model::back_propagation(int idx) {
	Layer* cur = nullptr;
	Layer* next = nullptr;
	this->target = target_set[idx];

	//output Layer
	cur = this->outputLayer;
	cur->dE_do = this->loss_diff(this->output, this->target);
	cur->do_dz = cur->activation_diff(cur->inputNeuron);
	cur->dz_dw = cur->preLayer->outputNeuron;
	cur->dE_dz = cur->dE_do * cur->do_dz;
	cur->dz_db.resize(cur->getNueronCnt(), 1);
	cur->dE_db = cur->dE_dz * cur->dz_db;
	for (int i = 0; i < this->nOutput; i++) {
		for (int j = 0; j < cur->preLayer->getNueronCnt(); j++) {
			cur->dE_dw(i, j) = cur->dE_do[i] * cur->do_dz[i] * cur->dz_dw[j];
			cur->weight(i, j) -= (this->learning_rate * cur->dE_dw(i, j));
		}
	}

	cur->bias -= (cur->dE_db * this->learning_rate);
	next = cur;
	cur = cur->preLayer; //dE_dh = sigma(dE_Oi)
	//hidden Layer
	do {
		for (int i = 0; i < cur->getNueronCnt(); i++) {
			float temp = 0;
			for (int j = 0; j < next->getNueronCnt(); j++) {
				temp += next->dE_dz[j] * next->weight(j, i);
			}
			cur->dE_do[i] = temp;
		}
		cur->do_dz = cur->activation_diff(cur->inputNeuron);
		cur->dz_dw = cur->preLayer->outputNeuron;
		cur->dE_dz = cur->dE_do * cur->do_dz;
		cur->dz_db.resize(cur->getNueronCnt(), 1);
		cur->dE_db = cur->dE_dz * cur->dz_db;
		//gradient
		for (int i = 0; i < cur->getNueronCnt(); i++) {
			for (int j = 0; j < cur->preLayer->getNueronCnt(); j++) {
				cur->dE_dw(i, j) = cur->dE_do[i] * cur->do_dz[i] * cur->dz_dw[j];
				//std::cout<<cur->dE_dw(j, k)<<'\t';
				cur->weight(i, j) -= (cur->dE_dw(i, j) * this->learning_rate);
			}
			//std::cout<<'\n';
		}
		cur->bias -= (cur->dE_db * this->learning_rate);
		next = cur;
		cur = cur->preLayer;
	} while (cur->preLayer != nullptr);

}

void Model::fit(int _total, int _epoch, int _batch) {
	this->init();
	this->total = _total;
	this->epoch = _epoch;
	this->batch = _batch;
	for (int i = 0; i < this->epoch; i++) {
		for (int j = 0; j < this->total; j++) {
			this->feed_forward(j);
			this->back_propagation(j);
		}
		std::cout << i + 1 << " epoch is done\n";
	}
}

void Model::evaluate(int _len,bool show) {
	this->accuracy = 0;
	float acc_sum = 0;
	float error_sum = 0;

	this->Layers[0]->preLayer = inputLayer;
	this->inputLayer->preLayer = nullptr;
	this->outputLayer->preLayer = this->Layers[this->depth - 1];
	for (int i = 0; i < _len; i++) {
		this->target = this->target_set[i];
		this->feed_forward(i);
		if (show) {
			std::cout << "\n\ntarget : \n";
			std::cout << this->target;
			std::cout << "output : " << this->getOutput() << '\n';
			std::cout << this->output;
		}
		if (this->check_success()) acc_sum += 1;
		error_sum += this->loss(output, target_set[i]);
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

void Model::setOutputFunction(Activation _activation) {
	this->outputLayer->setActivation(_activation);
}

void Model::addLayer(Layer* _layer) {
	this->Layers.push_back(_layer);
	this->depth += 1;
	if (this->Layers.size() == 1) {
		this->Layers[0]->preLayer = inputLayer;
		this->outputLayer->preLayer = Layers[0];
	}
	else {
		this->outputLayer->preLayer = Layers[Layers.size() - 1];
		this->Layers[Layers.size() - 1]->preLayer = Layers[Layers.size() - 2];
	}
	return;
}

void Model::addLayers(std::vector<Layer*>& _layers) {
	this->depth += _layers.size();
	for (Layer* layer : _layers) {
		this->Layers.push_back(layer);
		if (Layers.size() != 1) {
			Layers[Layers.size() - 1]->preLayer = Layers[Layers.size() - 2];
		}
	}
	if (Layers.size() == _layers.size()) {
		Layers[0]->preLayer = inputLayer;
	}
	else {
		outputLayer->preLayer = Layers[depth - 1];
		Layers[depth - 1]->preLayer = Layers[depth - 2];
	}
	outputLayer->preLayer = Layers[depth - 1];
	return;
}

void Model::setInput(std::vector<Vector<float>>& _input_set) {
	this->input_set = _input_set;
}

void Model::setTarget(std::vector<Vector<float>>& _target_set) {
	this->target_set = _target_set;

}

int Model::getOutput() {
	int idx = 0;
	float _max = -1e9;
	for (int i = 0; i < this->nOutput; i++) {
		if (this->output[i] > _max) {
			_max = this->output[i];
			idx = i;
		}
	}
	return idx + 1;
}

bool Model::check_success() {
	int ans = 0;
	for (int i = 0; i < nOutput; i++) {
		if (this->target[i] == 1) {
			ans = i + 1;
			break;
		}
	}
	return getOutput() == ans;
}

void Model::setLearningLate(float lr)
{
	this->learning_rate = lr;
}

#endif


