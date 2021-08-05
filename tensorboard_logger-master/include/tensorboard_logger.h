#ifndef TENSORBOARD_LOGGER_H
#define TENSORBOARD_LOGGER_H

#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "crc.h"
#include "event.pb.h"

using tensorflow::Event;
using tensorflow::Summary;

// extract parent dir from path by finding the last slash
std::string get_parent_dir(const std::string &path);

const std::string kProjectorConfigFile = "projector_config.pbtxt";
const std::string kProjectorPluginName = "projector";
const std::string kTextPluginName = "text";

class TensorBoardLogger {
   public:
    explicit TensorBoardLogger(const char *log_file) {
        bucket_limits_ = nullptr;
        ofs_ = new std::ofstream(
            log_file, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!ofs_->is_open())
            throw std::runtime_error("failed to open log_file " +
                                     std::string(log_file));
        log_dir_ = get_parent_dir(log_file);
    }
    ~TensorBoardLogger() {
        ofs_->close();
        if (bucket_limits_ != nullptr) {
            delete bucket_limits_;
            bucket_limits_ = nullptr;
        }
    }
    int add_scalar(const std::string &tag, int step, double value);
    int add_scalar(const std::string &tag, int step, float value);

    // https://github.com/dmlc/tensorboard/blob/master/python/tensorboard/summary.py#L127
    template <typename T>
    int add_histogram(const std::string &tag, int step, const T *value,
                      size_t num) {
        if (bucket_limits_ == nullptr) {
            generate_default_buckets();
        }

        std::vector<int> counts(bucket_limits_->size(), 0);
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::lowest();
        double sum = 0.0;
        double sum_squares = 0.0;
        for (size_t i = 0; i < num; ++i) {
            T v = value[i];
            auto lb = std::lower_bound(bucket_limits_->begin(),
                                       bucket_limits_->end(), v);
            counts[lb - bucket_limits_->begin()]++;
            sum += v;
            sum_squares += v * v;
            if (v > max) {
                max = v;
            } else if (v < min) {
                min = v;
            }
        }

        auto *histo = new tensorflow::HistogramProto();
        histo->set_min(min);
        histo->set_max(max);
        histo->set_num(num);
        histo->set_sum(sum);
        histo->set_sum_squares(sum_squares);
        for (size_t i = 0; i < counts.size(); ++i) {
            if (counts[i] > 0) {
                histo->add_bucket_limit((*bucket_limits_)[i]);
                histo->add_bucket(counts[i]);
            }
        }

        auto *summary = new tensorflow::Summary();
        auto *v = summary->add_value();
        v->set_tag(tag);
        v->set_allocated_histo(histo);

        return add_event(step, summary);
    };

    template <typename T>
    int add_histogram(const std::string &tag, int step,
                      const std::vector<T> &values) {
        return add_histogram(tag, step, values.data(), values.size());
    };

    // metadata (such as display_name, description) of the same tag will be
    // stripped to keep only the first one.
    int add_image(const std::string &tag, int step,
                  const std::string &encoded_image, int height, int width,
                  int channel, const std::string &display_name = "",
                  const std::string &description = "");
    int add_images(const std::string &tag, int step,
                   const std::vector<std::string> &encoded_images, int height,
                   int width, const std::string &display_name = "",
                   const std::string &description = "");
    int add_audio(const std::string &tag, int step,
                  const std::string &encoded_audio, float sample_rate,
                  int num_channels, int length_frame,
                  const std::string &content_type,
                  const std::string &display_name = "",
                  const std::string &description = "");
    int add_text(const std::string &tag, int step, const char *text);

    // `tensordata` and `metadata` should be in tsv format, and should be
    // manually created before calling `add_embedding`
    //
    // `tensor_name` is mandated to differentiate tensors
    //
    // TODO add sprite image support
    int add_embedding(
        const std::string &tensor_name, const std::string &tensordata_path,
        const std::string &metadata_path = "",
        const std::vector<uint32_t> &tensor_shape = std::vector<uint32_t>(),
        int step = 1 /* no effect */);
    // write tensor to binary file
    int add_embedding(
        const std::string &tensor_name,
        const std::vector<std::vector<float>> &tensor,
        const std::string &tensordata_filename,
        const std::vector<std::string> &metadata = std::vector<std::string>(),
        const std::string &metadata_filename = "",
        int step = 1 /* no effect */);
    int add_embedding(
        const std::string &tensor_name, const float *tensor,
        const std::vector<uint32_t> &tensor_shape,
        const std::string &tensordata_filename,
        const std::vector<std::string> &metadata = std::vector<std::string>(),
        const std::string &metadata_filename = "",
        int step = 1 /* no effect */);

   private:
    int generate_default_buckets();
    int add_event(int64_t step, Summary *summary);
    int write(Event &event);

    std::string log_dir_;
    std::ofstream *ofs_;
    std::vector<double> *bucket_limits_;
};  // class TensorBoardLogger

#endif  // TENSORBOARD_LOGGER_H
