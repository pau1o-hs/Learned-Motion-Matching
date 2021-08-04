#include "CustomLib.hpp"

// C++ neural network model defined with value semantics
struct Net : torch::nn::Module
{
    Net(torch::Device device, int64_t n_feature, int64_t n_hidden1, int64_t n_hidden2, int64_t n_hidden3, int64_t n_output)
    {
        this->fc1     = register_module("fc1",     torch::nn::Linear(n_feature, n_hidden1));
        this->fc2     = register_module("fc2",     torch::nn::Linear(n_hidden1, n_hidden2));
        this->fc3     = register_module("fc3",     torch::nn::Linear(n_hidden2, n_hidden3));
        this->predict = register_module("predict", torch::nn::Linear(n_hidden3, n_output));

        this->fc1->to(device);
        this->fc2->to(device);
        this->fc3->to(device);
        this->predict->to(device);
    }

    torch::Tensor forward(torch::Tensor feature, bool getEncoded)
    {
        feature = torch::elu(this->fc1(feature));
        feature = torch::elu(this->fc2(feature));

        if (getEncoded)
            return feature;
    
        feature = torch::elu(this->fc3(feature));
        feature = this->predict(feature);
        return feature;
    }

    torch::nn::Linear fc1 = nullptr;
    torch::nn::Linear fc2 = nullptr;
    torch::nn::Linear fc3 = nullptr;
    torch::nn::Linear predict = nullptr;
};

using namespace std;

int main()
{
    auto start = chrono::high_resolution_clock::now();

    cout << "[Device: " << at::cuda::getDeviceProperties(0)->name << "]" << endl << endl;

    // VARIABLES
    const int64_t n_feature = 539;
    const int64_t n_hidden1 = 512;
    const int64_t n_hidden2 = 32;
    const int64_t n_hidden3 = 512;
    const int64_t n_output = 539;
    const int64_t batch_size = 32;

    // INIT NN
    auto model = Net(device, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output);
    model.to(device);

    // DATASET
    vector<torch::Tensor> x, xNorm;
    LoadData("Poses", x, false);

	// NORMALIZE DATA
    for (int i = 0; i < (int) x.size(); i++)
    {
        auto mean = x[i].mean(1, true);
        auto std  = x[i].std(1, true).reshape({x[i].size(0), 1});

        xNorm.push_back((x[i] - mean) / std);
    }

    auto dataset = CustomDataset(xNorm, xNorm, 1, 0).map(torch::data::transforms::Stack<>());

    // DATA LOADER
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
    (std::move(dataset), torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

    // OPTIMIZER
    torch::optim::Adam optimizer(model.parameters(), 0.001);
	torch::optim::StepLR scheduler(optimizer, 100, 0.99);

    std::cout << "Training Compressor...\n";

    for (size_t epoch = 1; epoch <= 50000; epoch++)
    {
        int batchId = 0;
        float epochLoss = 0;

        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader)
        {
            batch.data = batch.data.to(device);
            batch.target = batch.target.to(device);

            // Execute the model on the input data.
            torch::Tensor prediction = model.forward(batch.data, false);
            
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::nn::MSELoss()(prediction, batch.target);

            // Reset gradients.
            optimizer.zero_grad();

            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();

            // Update the parameters based on the calculated gradients.
            optimizer.step();
            scheduler.step();
            
            epochLoss += prediction.size(0) * loss.item<float>();
            
            // Output the loss and checkpoint every 100 batches.
            // if (++batchId % 9 == 0 && epoch % 5000 == 0)
            // {
            //     std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << std::endl;
            //     // Serialize your model periodically as a checkpoint.
            //     // torch::save(model, "Compressor.pt");
            // }
        }

        if (epoch % 5000 == 0)
        {
            cout << "Epoch: " << epoch << " | Loss: " << epochLoss << endl;
        }
    }

    // SET PRECISION
    // std::cout << std::fixed << std::setprecision(4);

    std::cout << "Finished\n";

	// SAVE WEIGHTS
	ofstream weights("/home/pau1o-hs/Documents/NNWeights/Compressor.txt");
    ofstream latents("/home/pau1o-hs/Documents/Database/LatentVariables.txt");
	torch::Tensor weightData;

	// FC1
	weightData = model.fc1->weight.cpu().detach().transpose(0, 1).flatten();
	for (int i = 0; i < (int) weightData.size(0); i++) weights << weightData[i].item<float>() << endl;

	weightData = model.fc1->bias.cpu().detach();
	for (int i = 0; i < (int) weightData.size(0); i++) weights << weightData[i].item<float>() << endl;

	// FC2
	weightData = model.fc2->weight.cpu().detach().transpose(0, 1).flatten();
	for (int i = 0; i < (int) weightData.size(0); i++) weights << weightData[i].item<float>() << endl;

	weightData = model.fc2->bias.cpu().detach();
	for (int i = 0; i < (int) weightData.size(0); i++) weights << weightData[i].item<float>() << endl;

    // FC3
	weightData = model.fc3->weight.cpu().detach().transpose(0, 1).flatten();
	for (int i = 0; i < (int) weightData.size(0); i++) weights << weightData[i].item<float>() << endl;

	weightData = model.fc3->bias.cpu().detach();
	for (int i = 0; i < (int) weightData.size(0); i++) weights << weightData[i].item<float>() << endl;

	// PREDICT
	weightData = model.predict->weight.cpu().detach().transpose(0, 1).flatten();
	for (int i = 0; i < (int) weightData.size(0); i++) weights << weightData[i].item<float>() << endl;

	weightData = model.predict->bias.cpu().detach();
	for (int i = 0; i < (int) weightData.size(0); i++) weights << weightData[i].item<float>() << endl;


    for (int k = 0; k < (int) xNorm.size(); k++)
    {
        torch::Tensor encode = model.forward(xNorm[k], true);
        cout << encode.size(0) << ' ' << encode.size(1) << endl;

        for (int i = 0; i < (int) encode.size(0); i++)
        {
            for (int j = 0; j < (int) encode.size(1); j++)
                latents << encode[i][j].item<float>() << ' ';
            
            latents << endl;
        }
    }

    // RUNTIME
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
    cout << "Runtime: " << (float) (duration.count() / 60.0f) << " minutes" << endl;
}