#include "tensorboard_logger.h"

#include <google/protobuf/text_format.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "event.pb.h"
#include "projector_config.pb.h"

using std::endl;
using std::ifstream;
using std::numeric_limits;
using std::ofstream;
using std::ostringstream;
using std::string;
using std::to_string;
using std::vector;
using google::protobuf::TextFormat;
using tensorflow::EmbeddingInfo;
using tensorflow::Event;
using tensorflow::HistogramProto;
using tensorflow::ProjectorConfig;
// using tensorflow::SpriteMetadata;
using tensorflow::Summary;
using tensorflow::SummaryMetadata;
using tensorflow::TensorProto;

// https://github.com/dmlc/tensorboard/blob/master/python/tensorboard/summary.py#L115
int TensorBoardLogger::generate_default_buckets() {
    if (bucket_limits_ == nullptr) {
        bucket_limits_ = new vector<double>;
        vector<double> pos_buckets, neg_buckets;
        double v = 1e-12;
        while (v < 1e20) {
            pos_buckets.push_back(v);
            neg_buckets.push_back(-v);
            v *= 1.1;
        }
        pos_buckets.push_back(std::numeric_limits<double>::max());
        neg_buckets.push_back(std::numeric_limits<double>::lowest());

        bucket_limits_->insert(bucket_limits_->end(), neg_buckets.rbegin(),
                               neg_buckets.rend());
        bucket_limits_->insert(bucket_limits_->end(), pos_buckets.begin(),
                               pos_buckets.end());
    }

    return 0;
}

int TensorBoardLogger::add_scalar(const string &tag, int step, double value) {
    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_simple_value(value);
    return add_event(step, summary);
}

int TensorBoardLogger::add_scalar(const string &tag, int step, float value) {
    return add_scalar(tag, step, static_cast<double>(value));
}

int TensorBoardLogger::add_image(const string &tag, int step,
                                 const string &encoded_image, int height,
                                 int width, int channel,
                                 const string &display_name,
                                 const string &description) {
    auto *meta = new SummaryMetadata();
    meta->set_display_name(display_name.empty() ? tag : display_name);
    meta->set_summary_description(description);

    auto *image = new Summary::Image();
    image->set_height(height);
    image->set_width(width);
    image->set_colorspace(channel);
    image->set_encoded_image_string(encoded_image);

    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_allocated_image(image);
    v->set_allocated_metadata(meta);
    return add_event(step, summary);
}

int TensorBoardLogger::add_images(
    const std::string &tag, int step,
    const std::vector<std::string> &encoded_images, int height, int width,
    const std::string &display_name, const std::string &description) {
    auto *plugin_data = new SummaryMetadata::PluginData();
    plugin_data->set_plugin_name("images");
    auto *meta = new SummaryMetadata();
    meta->set_display_name(display_name.empty() ? tag : display_name);
    meta->set_summary_description(description);
    meta->set_allocated_plugin_data(plugin_data);

    auto *tensor = new TensorProto();
    tensor->set_dtype(tensorflow::DataType::DT_STRING);
    tensor->add_string_val(to_string(width));
    tensor->add_string_val(to_string(height));
    for (const auto &image : encoded_images) tensor->add_string_val(image);

    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_allocated_tensor(tensor);
    v->set_allocated_metadata(meta);

    return add_event(step, summary);
}

int TensorBoardLogger::add_audio(const string &tag, int step,
                                 const string &encoded_audio, float sample_rate,
                                 int num_channels, int length_frame,
                                 const string &content_type,
                                 const string &display_name,
                                 const string &description) {
    auto *meta = new SummaryMetadata();
    meta->set_display_name(display_name.empty() ? tag : display_name);
    meta->set_summary_description(description);

    auto *audio = new Summary::Audio();
    audio->set_sample_rate(sample_rate);
    audio->set_num_channels(num_channels);
    audio->set_length_frames(length_frame);
    audio->set_encoded_audio_string(encoded_audio);
    audio->set_content_type(content_type);

    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_allocated_audio(audio);
    v->set_allocated_metadata(meta);
    return add_event(step, summary);
}

int TensorBoardLogger::add_text(const string &tag, int step, const char *text) {
    auto *plugin_data = new SummaryMetadata::PluginData();
    plugin_data->set_plugin_name(kTextPluginName);

    auto *meta = new SummaryMetadata();
    meta->set_allocated_plugin_data(plugin_data);

    auto *tensor = new TensorProto();
    tensor->set_dtype(tensorflow::DataType::DT_STRING);

    auto *str_val = tensor->add_string_val();
    *str_val = text;

    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag(tag);
    v->set_allocated_tensor(tensor);
    v->set_allocated_metadata(meta);

    return add_event(step, summary);
}

int TensorBoardLogger::add_embedding(const std::string &tensor_name,
                                     const std::string &tensordata_path,
                                     const std::string &metadata_path,
                                     const std::vector<uint32_t> &tensor_shape,
                                     int step) {
    auto *plugin_data = new SummaryMetadata::PluginData();
    plugin_data->set_plugin_name(kProjectorPluginName);
    auto *meta = new SummaryMetadata();
    meta->set_allocated_plugin_data(plugin_data);

    const auto &filename = log_dir_ + kProjectorConfigFile;
    auto *conf = new ProjectorConfig();

    // parse possibly existing config file
    ifstream fin(filename);
    if (fin.is_open()) {
        ostringstream ss;
        ss << fin.rdbuf();
        TextFormat::ParseFromString(ss.str(), conf);
        fin.close();
    }

    auto *embedding = conf->add_embeddings();
    embedding->set_tensor_name(tensor_name);
    embedding->set_tensor_path(tensordata_path);
    if (metadata_path != "") {
        embedding->set_metadata_path(metadata_path);
    }
    if (tensor_shape.size() > 0) {
        for (auto shape : tensor_shape) embedding->add_tensor_shape(shape);
    }

    // `conf` and `embedding` will be deleted by ProjectorConfig destructor

    ofstream fout(filename);
    string content;
    TextFormat::PrintToString(*conf, &content);
    fout << content;
    fout.close();

    // Following line is just to add plugin and does not hold any meaning
    auto *summary = new Summary();
    auto *v = summary->add_value();
    v->set_tag("embedding");
    v->set_allocated_metadata(meta);

    return add_event(step, summary);
}

int TensorBoardLogger::add_embedding(
    const std::string &tensor_name,
    const std::vector<std::vector<float>> &tensor,
    const std::string &tensordata_filename,
    const std::vector<std::string> &metadata,
    const std::string &metadata_filename, int step) {
    ofstream binary_tensor_file(log_dir_ + tensordata_filename, std::ios::binary);
    if (!binary_tensor_file.is_open()) {
        throw std::runtime_error("failed to open binary tensor file " +
                                 log_dir_ + tensordata_filename);
    }

    for (const auto &vec : tensor) {
        binary_tensor_file.write(reinterpret_cast<const char *>(vec.data()),
                                 vec.size() * sizeof(float));
    }
    binary_tensor_file.close();
    if (metadata.size() > 0) {
        if (metadata.size() != tensor.size()) {
            throw std::runtime_error("tensor size != metadata size");
        }
        ofstream metadata_file(log_dir_ + metadata_filename);
        if (!metadata_file.is_open()) {
            throw std::runtime_error("failed to open metadata file " +
                                     log_dir_ + metadata_filename);
        }
        for (const auto &meta : metadata) metadata_file << meta << endl;
        metadata_file.close();
    }
    vector<uint32_t> tensor_shape;
    tensor_shape.push_back(tensor.size());
    tensor_shape.push_back(tensor[0].size());
    return add_embedding(tensor_name, tensordata_filename, metadata_filename,
                         tensor_shape, step);
}

int TensorBoardLogger::add_embedding(const std::string &tensor_name,
                                     const float *tensor,
                                     const std::vector<uint32_t> &tensor_shape,
                                     const std::string &tensordata_filename,
                                     const std::vector<std::string> &metadata,
                                     const std::string &metadata_filename,
                                     int step) {
    ofstream binary_tensor_file(log_dir_ + tensordata_filename, std::ios::binary);
    if (!binary_tensor_file.is_open()) {
        throw std::runtime_error("failed to open binary tensor file " +
                                 log_dir_ + tensordata_filename);
    }

    uint32_t num_elements = 1;
    for (auto shape : tensor_shape) num_elements *= shape;
    binary_tensor_file.write(reinterpret_cast<const char *>(tensor),
                             num_elements * sizeof(float));
    binary_tensor_file.close();
    if (metadata.size() > 0) {
        if (metadata.size() != tensor_shape[0]) {
            throw std::runtime_error("tensor size != metadata size");
        }
        ofstream metadata_file(log_dir_ + metadata_filename);
        if (!metadata_file.is_open()) {
            throw std::runtime_error("failed to open metadata file " +
                                     log_dir_ + metadata_filename);
        }
        for (const auto &meta : metadata) metadata_file << meta << endl;
        metadata_file.close();
    }
    return add_embedding(tensor_name, tensordata_filename, metadata_filename,
                         tensor_shape, step);
}

int TensorBoardLogger::add_event(int64_t step, Summary *summary) {
    Event event;
    double wall_time = time(nullptr);
    event.set_wall_time(wall_time);
    event.set_step(step);
    event.set_allocated_summary(summary);
    return write(event);
}

int TensorBoardLogger::write(Event &event) {
    string buf;
    event.SerializeToString(&buf);
    auto buf_len = static_cast<uint64_t>(buf.size());
    uint32_t len_crc =
        masked_crc32c((char *)&buf_len, sizeof(buf_len));  // NOLINT
    uint32_t data_crc = masked_crc32c(buf.c_str(), buf.size());

    ofs_->write((char *)&buf_len, sizeof(buf_len));  // NOLINT
    ofs_->write((char *)&len_crc, sizeof(len_crc));  // NOLINT
    ofs_->write(buf.c_str(), buf.size());
    ofs_->write((char *)&data_crc, sizeof(data_crc));  // NOLINT
    ofs_->flush();
    return 0;
}

string get_parent_dir(const string &path) {
    auto last_slash_pos = path.find_last_of("/\\");
    if (last_slash_pos == string::npos) {
        return "./";
    }
    return path.substr(0, last_slash_pos + 1);
}
