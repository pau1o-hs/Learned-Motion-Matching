#include "CustomLib.hpp"

void LoadData(string fileName, vector<torch::Tensor>& toTensor, bool combineLatent)
{
	ifstream fileIn("/home/pau1o-hs/Documents/Database/" + fileName + ".txt");
	ifstream latentVars("/home/pau1o-hs/Documents/Database/LatentVariables.txt");

	string s;
	int clip = 0;

	vector<vector<vector<float>>> database;
	database.push_back(vector<vector<float>>());

	//read a line into 's' from 'fileIn' each time
	for (int i = 0; getline(fileIn, s); i++)
	{
		// END OF CLIP
		if (s == "")
		{
			i = -1, clip++;
			database.push_back(vector<vector<float>>());

			continue;
		}

		database[clip].push_back(vector<float>());

		//use the string 's' as input stream, the usage of 'sin' is just like 'cin'
		istringstream sin1(s);
		float value;
		
		while (sin1 >> value) 
			database[clip][i].push_back(value);

		// COMBINING WITH LATENT VARIABLES
		if (combineLatent)
		{
			getline(latentVars, s);
			istringstream sin2(s);

			while (sin2 >> value)
				database[clip][i].push_back(value);
		}
	}

	// Copying into a tensor
	auto options = torch::TensorOptions().dtype(torch::kFloat);
	int m = (int) database[0][0].size();

	for (int i = 0; i < clip; i++)
	{
		int n = (int) database[i].size();

		toTensor.push_back(torch::zeros({ n, m }, options).to(device));

		for (int j = 0; j < n; j++)
			toTensor[i].slice(0, j, j + 1) = torch::from_blob(database[i][j].data(), {m}, options).clone();
	}

	return;
}