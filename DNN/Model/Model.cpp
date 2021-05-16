#ifndef VANILLA_DNN_MODEL_CPP
#define VANILLA_DNN_MODEL_CPP

#include "Model.hpp"


Model::Model() {

}

Model::Model(int _nInput, int _nOutput) {
	this->depth = 0;
	this->nInput = _nInput;
	this->nOutput = _nOutput;
	this->inputLayer = new Layer(_nInput);
	this->outputLayer = new Layer(_nOutput);
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
		cur->weight.resize(cur->getNueronCnt(), cur->preLayer->getNueronCnt(), ((float)rand() / (RAND_MAX)));
		cur->dE_dw.resize(cur->getNueronCnt(), cur->preLayer->getNueronCnt(), 0);
		cur->dz_dw.resize(cur->preLayer->getNueronCnt());
		cur = cur->preLayer;
	} while (cur->preLayer != nullptr);
}

void Model::feed_forward(int idx) {
	this->inputLayer->inputNeuron = this->input_set[idx];
	this->inputLayer->outputNeuron = this->inputLayer->inputNeuron;
	for (Layer* layer : Layers) {
		layer->inputNeuron = (layer->weight * layer->preLayer->outputNeuron);// + layer->bias;
		layer->outputNeuron = layer->activation(layer->inputNeuron);
	}
	Layer* temp = this->outputLayer;
	temp->inputNeuron = temp->weight * this->Layers[this->depth - 1]->outputNeuron;
	temp->outputNeuron = temp->activation(temp->inputNeuron);
	this->output = this->outputLayer->outputNeuron;
	std::cout << "final input\n";
	std::cout << this->outputLayer->inputNeuron;
	std::cout << "final output\n";
	std::cout << this->outputLayer->outputNeuron;
	return;
}

void Model::back_propagation(int idx) {
	Layer* cur = nullptr;
	Layer* next = nullptr;
	this->input = input_set[idx];
	this->target = target_set[idx];
	//output Layer
	cur = this->outputLayer;
	cur->dE_do = this->loss_diff(this->output, this->target);
	cur->do_dz = cur->activation_diff(cur->inputNeuron);
	cur->dz_dw = cur->preLayer->outputNeuron;
	cur->dE_dz = cur->dE_do * cur->do_dz; //for chain rule
	for (int j = 0; j < this->nOutput; j++) {
		for (int k = 0; k < cur->preLayer->getNueronCnt(); k++) {
			cur->dE_dw(j, k) = cur->dE_do[j] * cur->do_dz[j] * cur->dz_dw[k];
			cur->weight(j, k) -= cur->dE_dw(j, k);
		}
	}

	next = cur;
	cur = cur->preLayer; //dE_dh = sigma(dE_Oi)
	//hidden Layer
	do {
		for (int j = 0; j < cur->getNueronCnt(); j++) {
			float temp = 0;
			for (int k = 0; k < next->getNueronCnt(); k++) {
				temp += next->dE_dz[k] * next->weight(k, j);
			}
			cur->dE_do[j] = temp;
		}
		cur->do_dz = cur->activation_diff(cur->inputNeuron);
		cur->dz_dw = cur->preLayer->outputNeuron;
		cur->dE_dz = cur->dE_do * cur->do_dz;

		//gradient
		for (int j = 0; j < cur->getNueronCnt(); j++) {
			for (int k = 0; k < cur->preLayer->getNueronCnt(); k++) {
				cur->dE_dw(j, k) = cur->dE_do[j] * cur->do_dz[j] * cur->dz_dw[k];
				//std::cout<<cur->dE_dw(j, k)<<'\t';
				cur->weight(j, k) -= cur->dE_dw(j, k);
			}
			//std::cout<<'\n';
		}
		next = cur;
		cur = cur->preLayer;
	} while (cur->preLayer != nullptr);

}

void Model::fit(int _batch, int _epoch) {
	this->init();
	this->batch = _batch;
	this->epoch = _epoch;
	for (int i = 0; i < this->epoch; i++) {
		for (int j = 0; j < this->batch; j++) {
			this->feed_forward(j);
			this->back_propagation(j);
		}
	}
}

void Model::evaluate(int _batch) {
	this->accuracy = 0;
	float acc_sum = 0;
	float error_sum = 0;
	this->batch = _batch;
	for (int i = 0; i < this->batch; i++) {
		feed_forward(i);
		if (check_success(i)) acc_sum += 1;
		error_sum += this->loss(output, target_set[i]);
	}
	this->accuracy = acc_sum / _batch;
	this->error = error_sum / _batch;
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
	int idx = 0, _max = -1e9;
	for (int i = 0; i < this->nOutput; i++) {
		if (this->output[i] > _max) {
			_max = this->output[i];
			idx = i;
		}
	}
	return idx + 1;
}

bool Model::check_success(int idx) {
	int ans = 0;
	for (int i = 0; i < nOutput; i++) {
		if (target_set[idx][i] != 0) {
			ans = idx + 1;
			break;
		}
	}
	return getOutput() == ans;
}

#endif


