#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "tensorboard_logger.h"

using namespace std;

string read_binary_file(const string filename) {
    ostringstream ss;
    ifstream fin(filename, ios::binary);
    if (!fin) {
        std::cerr << "failed to open file " << filename << std::endl;
        return "";
    }
    ss << fin.rdbuf();
    fin.close();
    return ss.str();
}

int test_log_scalar(TensorBoardLogger& logger) {
    cout << "test log scalar" << endl;
    default_random_engine generator;
    normal_distribution<double> default_distribution(0, 1.0);
    for (int i = 0; i < 10; ++i) {
        logger.add_scalar("scalar", i, default_distribution(generator));
    }

    return 0;
}

int test_log_histogram(TensorBoardLogger& logger) {
    cout << "test log histogram" << endl;
    default_random_engine generator;
    for (int i = 0; i < 10; ++i) {
        normal_distribution<double> distribution(i * 0.1, 1.0);
        vector<float> values;
        for (int j = 0; j < 10000; ++j)
            values.push_back(distribution(generator));
        logger.add_histogram("histogram", i, values);
    }

    return 0;
}

int test_log_image(TensorBoardLogger& logger) {
    cout << "test log image" << endl;
    // read images
    auto image1 = read_binary_file("./assets/text.png");
    auto image2 = read_binary_file("./assets/audio.png");

    // add single image
    logger.add_image("TensorBoard Text Plugin", 1, image1, 1864, 822, 3,
                     "TensorBoard", "Text");
    logger.add_image("TensorBoard Audo Plugin", 1, image2, 1766, 814, 3,
                     "TensorBoard", "Audio");

    // add multiple images
    // FIXME This seems doesn't work anymore.
    // logger.add_images(
    //     "Multiple Images", 1, {image1, image2}, 1502, 632, "test", "not
    //     working");

    return 0;
}

int test_log_audio(TensorBoardLogger& logger) {
    cout << "test log audio" << endl;
    auto audio = read_binary_file("./assets/file_example_WAV_1MG.wav");
    logger.add_audio("Audio Sample", 1, audio, 8000, 2, 8000 * 16 * 2 * 33,
                     "audio/wav", "Impact Moderato",
                     "https://file-examples.com/index.php/sample-audio-files/"
                     "sample-wav-download/");

    return 0;
}

int test_log_text(TensorBoardLogger& logger) {
    cout << "test log text" << endl;
    logger.add_text("Text Sample", 1, "Hello World");

    return 0;
}

int test_log_embedding(TensorBoardLogger& logger) {
    cout << "test log embedding" << endl;
    // test add embedding
    logger.add_embedding("vocab", "../assets/vecs.tsv", "../assets/meta.tsv");
    logger.add_embedding("another vocab without labels", "../assets/vecs.tsv");

    // test add binary embedding
    vector<vector<float>> tensor;
    string line;
    ifstream vec_file("assets/vecs.tsv");
    uint32_t num_elements = 1;
    while (getline(vec_file, line)) {
        istringstream values(line);
        vector<float> vec;
        copy(istream_iterator<float>(values), istream_iterator<float>(),
             back_inserter(vec));
        num_elements += vec.size();
        tensor.push_back(vec);
    }
    vec_file.close();

    vector<string> meta;
    ifstream meta_file("assets/meta.tsv");
    while (getline(meta_file, line)) {
        meta.push_back(line);
    }
    meta_file.close();
    logger.add_embedding("binary tensor", tensor, "tensor.bin", meta,
                         "binary_tensor.tsv");

    // test tensor stored as 1d array
    float* tensor_1d = new float[num_elements];
    for (size_t i = 0; i < tensor.size(); i++) {
        const auto& vec = tensor[i];
        memcpy(tensor_1d + i * vec.size(), vec.data(),
               vec.size() * sizeof(float));
    }
    vector<uint32_t> tensor_shape;
    tensor_shape.push_back(tensor.size());
    tensor_shape.push_back(tensor[0].size());
    logger.add_embedding("binary tensor 1d", tensor_1d, tensor_shape,
                         "tensor_1d.bin", meta, "binary_tensor_1d.tsv");
    delete[] tensor_1d;

    return 0;
}

int test_log(const char* log_file) {
    TensorBoardLogger logger(log_file);

    test_log_scalar(logger);
    test_log_histogram(logger);
    test_log_image(logger);
    test_log_audio(logger);
    test_log_text(logger);
    test_log_embedding(logger);

    return 0;
}

int main(int argc, char* argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    int ret = test_log("./demo/tfevents.pb");
    assert(ret == 0);

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
