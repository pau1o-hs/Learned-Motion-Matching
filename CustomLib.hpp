#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <time.h>

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/autograd/autograd.h>

using namespace std;

// DEVICE
static torch::Device device(torch::kCUDA);

void LoadData(string fileName, vector<torch::Tensor>& toTensor, bool combineLatent);

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
	// Declare 2 vectors of tensors for images and labels
	vector<torch::Tensor> data;
	vector<torch::Tensor> target;

	int dataSize;
	int windowSize;
	float gaussianNoise;

public:
	// Constructor
	explicit CustomDataset(vector<torch::Tensor> data, vector<torch::Tensor> target, int windowSize, float gaussianNoise)
	{
		this->data   = data;
		this->target = target;

		this->windowSize = windowSize;
		this->gaussianNoise = gaussianNoise;
		this->dataSize = 0;

		for (int i = 0; i < (int) data.size(); i++)
			this->dataSize += (int) data[i].size(0);
		
		srand (time(NULL));
	};

	// Override get() function to return tensor at location index
	torch::data::Example<> get(size_t defautRand) override
	{
		torch::Tensor dataWindow   = torch::zeros({windowSize, data[0].size(1)});
		torch::Tensor targetWindow = torch::zeros({windowSize, target[0].size(1)});

		int clip  = rand() % ((int) data.size());
		int index = rand() % ((int) data[clip].size(0) - (this->windowSize - 1));
		
		for (int i = 0; i < windowSize; i++)
		{
			int k = (index + i) % this->data[clip].size(0);

			dataWindow[i]   = this->data[clip][k];
			targetWindow[i] = this->target[clip][k];
		}

		if (this->gaussianNoise == 0)
			return { dataWindow, targetWindow };
		
		torch::Tensor rand  = torch::randn_like(dataWindow);
		torch::Tensor range = torch::rand_like(dataWindow) * this->gaussianNoise;

		torch::Tensor noise = rand * range * dataWindow;
		dataWindow = dataWindow + noise;

		// cout << noise << endl;
		// cout << "Feature: " << dataWindow << endl;
		// cout << "Target: " << targetWindow << endl;

		return { dataWindow, targetWindow };
	};

	// Return the length of data
	torch::optional<size_t> size() const override
	{
		return this->dataSize;
	};
};