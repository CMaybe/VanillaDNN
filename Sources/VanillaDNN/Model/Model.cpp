#ifndef VANILLA_DNN_MODEL_CPP
#define VANILLA_DNN_MODEL_CPP

#include <VanillaDNN/Model/Model.hpp>


Model::Model(int _nInput, int _nOutput) {
	this->depth = 0;
	this->nInput = _nInput;
	this->nOutput = _nOutput;
	this->inputLayer = new Layer(_nInput);
	this->outputLayer = new OutputLayer(_nOutput);
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
	for (Layer* layer : Layers) {
		if (layer != nullptr) delete layer;
	}
	Layers.clear();
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
	this->Layers[0]->preLayer = inputLayer;
	this->inputLayer->preLayer = nullptr;
	this->inputLayer->inputNeuron.resize(this->batch_size);
	this->inputLayer->outputNeuron.resize(this->batch_size);
	this->outputLayer->preLayer = this->Layers[this->depth - 1];
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
		batch_dE_do.insert(batch_dE_do.begin(), Vector<float>(cur->getNueronCnt(), 0));
		batch_do_dz.insert(batch_do_dz.begin(), Vector<float>(cur->getNueronCnt(), 0));
		batch_dz_dw.insert(batch_dz_dw.begin(), Vector<float>(cur->preLayer->getNueronCnt(),0));
		batch_dE_dz.insert(batch_dE_dz.begin(), Vector<float>(cur->getNueronCnt(), 0));
		batch_dE_db.insert(batch_dE_db.begin(), Vector<float>(cur->getNueronCnt(), 0));
		batch_dz_db.insert(batch_dz_db.begin(), Vector<float>(cur->getNueronCnt(), 0));
		

		cur = cur->preLayer;
	} while (cur->preLayer != nullptr);
}

void Model::feed_forward(int idx) {
	this->inputLayer->inputNeuron[idx % this->batch_size] = this->input_set[idx];
	this->inputLayer->outputNeuron[idx % this->batch_size] = this->input_set[idx];;
	for (Layer* layer : Layers) {
		layer->inputNeuron[idx % this->batch_size] = (layer->weight * layer->preLayer->outputNeuron[idx % this->batch_size]) + layer->bias;
		layer->outputNeuron[idx % this->batch_size] = layer->activation(layer->inputNeuron[idx % this->batch_size]);
	}
	Layer* cur = this->outputLayer;
	cur->inputNeuron[idx % this->batch_size] = (cur->weight * cur->preLayer->outputNeuron[idx % this->batch_size]) + cur->bias;
	cur->outputNeuron[idx % this->batch_size] = cur->activation(cur->inputNeuron[idx % this->batch_size]);
	this->output_set[idx % this->batch_size] = cur->outputNeuron[idx % this->batch_size];
	return;
}

void Model::back_propagation(int idx) {
	Layer* cur = nullptr;
	Layer* next = nullptr;
	int _depth = this->depth;
	//output Layer
	cur = this->outputLayer;
	cur->dE_do[idx % this->batch_size] = this->loss_diff(this->output_set[idx % this->batch_size], this->target_set[idx]);
	cur->do_dz[idx % this->batch_size] = cur->activation_diff(cur->inputNeuron[idx % this->batch_size]);
	cur->dz_dw[idx % this->batch_size] = cur->preLayer->outputNeuron[idx % this->batch_size];
	cur->dE_dz[idx % this->batch_size] = cur->dE_do[idx % this->batch_size] * cur->do_dz[idx % this->batch_size];
	cur->dz_db[idx % this->batch_size].resize(cur->getNueronCnt(), 1);
	cur->dE_db[idx % this->batch_size] = cur->dE_dz[idx % this->batch_size] * cur->dz_db[idx % this->batch_size];
	
	this->batch_dE_do[_depth] += cur->dE_do[idx % this->batch_size];
	this->batch_do_dz[_depth] += cur->do_dz[idx % this->batch_size];
	this->batch_dz_dw[_depth] += cur->dz_dw[idx % this->batch_size];
	this->batch_dE_dz[_depth] += cur->dE_dz[idx % this->batch_size];
	this->batch_dz_db[_depth] += cur->dz_db[idx % this->batch_size];
	this->batch_dE_db[_depth] += cur->dE_db[idx % this->batch_size];
	
	for (int i = 0; i < this->nOutput; i++) {
		for (int j = 0; j < cur->preLayer->getNueronCnt(); j++) {
			cur->dE_dw[idx % this->batch_size](i, j) = 
				cur->dE_do[idx % this->batch_size][i] * cur->do_dz[idx % this->batch_size][i] * cur->dz_dw[idx % this->batch_size][j];
			this->batch_dE_dw[_depth](i,j) += cur->dE_dw[idx % this->batch_size](i, j);
		}
	}
	
	_depth -= 1;
	next = cur;
	cur = cur->preLayer; 
	//hidden Layer
	do {
		for (int i = 0; i < cur->getNueronCnt(); i++) {
			float temp = 0;
			for (int j = 0; j < next->getNueronCnt(); j++) {
				temp += next->dE_dz[idx % this->batch_size][j] * next->weight(j, i);
			}
			cur->dE_do[idx % this->batch_size][i] = temp;
		}
		cur->do_dz[idx % this->batch_size] = cur->activation_diff(cur->inputNeuron[idx % this->batch_size]);
		cur->dz_dw[idx % this->batch_size] = cur->preLayer->outputNeuron[idx % this->batch_size];
		cur->dE_dz[idx % this->batch_size] = cur->dE_do[idx % this->batch_size] * cur->do_dz[idx % this->batch_size];
		cur->dz_db[idx % this->batch_size].resize(cur->getNueronCnt(), 1);
		cur->dE_db[idx % this->batch_size] = cur->dE_dz[idx % this->batch_size] * cur->dz_db[idx % this->batch_size];
		
		this->batch_dE_do[_depth] += cur->dE_do[idx % this->batch_size];
		this->batch_do_dz[_depth] += cur->do_dz[idx % this->batch_size];
		this->batch_dz_dw[_depth] += cur->dz_dw[idx % this->batch_size];
		this->batch_dE_dz[_depth] += cur->dE_dz[idx % this->batch_size];
		this->batch_dz_db[_depth] += cur->dz_db[idx % this->batch_size];
		this->batch_dE_db[_depth] += cur->dE_db[idx % this->batch_size];
		
		//gradient
		for (int i = 0; i < cur->getNueronCnt(); i++) {
			for (int j = 0; j < cur->preLayer->getNueronCnt(); j++) {
				cur->dE_dw[idx % this->batch_size](i, j) = 
					cur->dE_do[idx % this->batch_size][i] * cur->do_dz[idx % this->batch_size][i] * cur->dz_dw[idx % this->batch_size][j];
				this->batch_dE_dw[_depth](i,j) += cur->dE_dw[idx % this->batch_size](i, j);
			}
			//std::cout<<'\n';
		}
		next = cur;
		cur = cur->preLayer;
		_depth -= 1;
	} while (cur->preLayer != nullptr);
}

void Model::update(){
	Layer* cur = nullptr;
	int _depth = this->depth;
	//output Layer
	cur = this->outputLayer;
	cur->weight -= this->optimizer->getWeightGradient(this->batch_dE_dw[_depth], _depth) / this->batch_size;
	cur->bias -= this->batch_dE_db[_depth]  / this->batch_size;
	cur = cur->preLayer; //dE_dh = sigma(dE_Oi)
	_depth -= 1;

	//hidden Layer
	do {
		//gradient
		cur->weight -= this->optimizer->getWeightGradient(this->batch_dE_dw[_depth], _depth) / this->batch_size;
		cur->bias -= this->batch_dE_db[_depth]  / this->batch_size;
		_depth -= 1;
		cur = cur->preLayer;
	} while (cur->preLayer != nullptr);
	
	
	// init batch data
	
	batch_dE_dw.clear();
	batch_dE_do.clear();
	batch_do_dz.clear();
	batch_dz_dw.clear();
	batch_dE_dz.clear();
	batch_dE_db.clear();
	batch_dz_db.clear();
	
	cur = this->outputLayer;
	do {
		
		batch_dE_dw.insert(batch_dE_dw.begin(), Matrix<float>(cur->getNueronCnt(), cur->preLayer->getNueronCnt(), 0));
		batch_dE_do.insert(batch_dE_do.begin(), Vector<float>(cur->getNueronCnt(), 0));
		batch_do_dz.insert(batch_do_dz.begin(), Vector<float>(cur->getNueronCnt(), 0));
		batch_dz_dw.insert(batch_dz_dw.begin(), Vector<float>(cur->preLayer->getNueronCnt(),0));
		batch_dE_dz.insert(batch_dE_dz.begin(), Vector<float>(cur->getNueronCnt(), 0));
		batch_dE_db.insert(batch_dE_db.begin(), Vector<float>(cur->getNueronCnt(), 0));
		batch_dz_db.insert(batch_dz_db.begin(), Vector<float>(cur->getNueronCnt(), 0));
		
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
	this->Layers[0]->preLayer = inputLayer;
	this->inputLayer->preLayer = nullptr;
	this->outputLayer->preLayer = this->Layers[this->depth - 1];
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


