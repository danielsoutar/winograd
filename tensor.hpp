// tensor.hpp - Header file for Tensor object
// Created by: Daniel Soutar
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

struct TensorSizeStruct {
public:
  std::vector<unsigned int> shape;

  TensorSizeStruct() {}

  TensorSizeStruct(std::vector<unsigned int> s) { shape = s; }

  unsigned int &operator[](int i) { return shape[i]; }

  std::vector<unsigned int> get_dimensions() { return shape; }

  unsigned int get_total_size() {
    unsigned int total_size = 1;
    for (auto &size : shape)
      total_size *= size;
    return total_size;
  };
};

class Conv2DParamPack {
public:
  // Assume square inputs and filters.
  Conv2DParamPack(std::vector<unsigned int> params) {
    stride_ = params[0];
    padding_ = params[1];
    fil_size_ = params[2];
  }

  TensorSizeStruct get_out_sizes(TensorSizeStruct inp_sz,
                                 TensorSizeStruct fil_sz) {
    unsigned int out_size =
        std::floor((inp_sz[2] + 2 * padding_ - fil_sz[2]) / stride_) + 1;
    std::vector<unsigned int> out_shape = {inp_sz[0], fil_sz[0], out_size,
                                           out_size};
    return out_shape;
  }

  const unsigned int stride() { return stride_; }
  const unsigned int padding() { return padding_; }
  const unsigned int fil_size() { return fil_size_; }

private:
  unsigned int stride_, padding_, fil_size_;
};

// forward-declare tensor class
class Tensor {
public:
  Tensor(TensorSizeStruct sz) {
    shape_ = sz;
    data_.resize(shape_.get_total_size());
    std::iota(data_.begin(), data_.end(), 0.0f);
  }

  Tensor(std::vector<float> data, TensorSizeStruct sz) {
    data_ = data;
    shape_ = sz;
  }

  Tensor(Tensor a, Tensor b, Conv2DParamPack params) {
    shape_ = params.get_out_sizes(a.shape(), b.shape());
    data_.resize(shape_.get_total_size());
  }

  std::vector<float> data() { return data_; }

  void set_data(std::vector<float> data) { data_ = data; }
  TensorSizeStruct shape() { return shape_; }

  void print() {
    auto dims = shape_.get_dimensions();
    for (int n = 0; n < dims[0]; ++n) {
      for (int c = 0; c < dims[1]; ++c) {
        for (int h = 0; h < dims[2]; ++h) {
          for (int w = 0; w < dims[3]; ++w)
            std::cout << std::setw(4)
                      << data_[n * dims[1] * dims[2] * dims[3] +
                               c * dims[2] * dims[3] + h * dims[3] + w]
                      << " ";
          std::cout << "\n";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  void print(int c) {
    auto dims = shape_.get_dimensions();
    for (int h = 0; h < dims[2]; ++h) {
      for (int w = 0; w < dims[3]; ++w) {
        std::cout << std::setw(2) << data_[c * dims[1] + h * dims[2] + w]
                  << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  unsigned int shape(unsigned int i) { return shape_[i]; }

private:
  std::vector<float> data_;
  TensorSizeStruct shape_;
};
