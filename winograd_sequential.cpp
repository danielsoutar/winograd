// winograd_sequential.cpp - Implementation of the Winograd transform for convolutions
// Created by: Daniel Soutar
#include "tensor.hpp"
#include <iostream>
#include <vector>

/**
 * The Winograd algorithm here performs a 2D convolution on a 4-dimensional input.
 * The input data is assumed to be in NCHW format, where:
 * 
 *  > N denotes the minibatch size
 *  > C denotes the number of input channels or feature maps
 *  > H denotes the input's height
 *  > W denotes the input's width
 *
 * The filter is assumed to be in FCHW format, where:
 *
 *  > F denotes the number of output channels or feature maps
 *  > C denotes the number of input channels or feature maps (must match the input)
 *  > H denotes the filter's height
 *  > W denotes the filter's width
 *
 * Dilation > 1 is not supported. This operation is only supported for stride=1 for
 * performance reasons.
 */

/**
 * Transform the input for the Winograd algorithm. This means taking in a 4x4
 * slice of the input containing 4 3x3 slices to do convolutions on, and
 * returning a Cx4x4 output, where C is the number of input channels.
 *
 * In our case, we take the (i,j) 4x4 slice. This is done in row-major order.
 */
std::vector<float> transform_input(std::vector<float> input, int channels, int height, int width, int padding, int tile_i, int tile_j) {
  std::vector<float> tile(4 * 4 * channels);
  int hw = height * width;
  int r = (0 - padding) + tile_i * 2;
  int c = (0 - padding) + tile_j * 2;
  int tile_size = 4;
  for(int channel = 0; channel < channels; ++channel) {
    int channel_offset = channel * hw;
    int channel_tile_offset = channel * 16;
    for(int input_i = r; input_i < r + tile_size; ++input_i) {
      for(int input_j = c; input_j < c + tile_size; ++input_j) {
        if(input_i > -1 && input_i < height && input_j > -1 && input_j < width)
          tile[channel_tile_offset + (input_i - r) * 4 + input_j]
            = input[channel_offset + input_i * width + input_j];
        else
          tile[channel_tile_offset + (input_i - r) * 4 + input_j] = 0.0f;
      }
    }
  }

  // Now have the slices of the input. For each channel, perform the input transform.
  // TIME TO DO ADDITION
#ifndef TILE
#define TILE(X) tile[channel_tile_offset + X]
  for(int channel = 0; channel < channels; ++channel) {
    int channel_tile_offset = channel * 16;
    int bd0 = TILE(0) - TILE(8);
    int bd1 = TILE(1) - TILE(9);
    int bd2 = TILE(2) - TILE(10);
    int bd3 = TILE(3) - TILE(11);

    TILE(0) = bd0 - bd2;
    TILE(1) = bd1 + bd2;
    TILE(2) = bd2 - bd1;
    TILE(3) = bd1 - bd3;

    int bd4 = TILE(4) + TILE(8);
    int bd5 = TILE(5) + TILE(9);
    int bd6 = TILE(6) + TILE(10);
    int bd7 = TILE(7) + TILE(11);

    TILE(4) = bd4 - bd6;
    TILE(5) = bd5 + bd6;
    TILE(6) = bd6 - bd5;
    TILE(7) = bd5 - bd7;

    int bd8 = TILE(8) - TILE(4);
    int bd9 = TILE(9) - TILE(5);
    int bd10 = TILE(10) - TILE(6);
    int bd11 = TILE(11) - TILE(7);

    TILE(8) = bd8 - bd10;
    TILE(9) = bd9 + bd10;
    TILE(10) = bd10 - bd9;
    TILE(11) = bd9 - bd11;

    int bd12 = TILE(4) - TILE(12);
    int bd13 = TILE(5) - TILE(15);
    int bd14 = TILE(6) - TILE(14);
    int bd15 = TILE(7) - TILE(13);

    TILE(12) = bd12 - bd14;
    TILE(13) = bd13 + bd14;
    TILE(14) = bd14 - bd13;
    TILE(15) = bd13 - bd15;
  }
#undef TILE
#endif

  return tile;
}

// Perform a 2D convolution using the Winograd transform
Tensor conv2d(Tensor inp, Tensor fil, Conv2DParamPack conv2d_params) {
  // Validate inputs - only for 3x3 filters with stride 1.
  // static_assert(conv2d_params.fil_size() == 3, "Can only use Winograd for 3x3 filters");
  // static_assert(conv2d_params.stride() == 1, "Can only use Winograd for unit stride");

  // Initialise an output tensor based on inputs and operation.
  Tensor out(inp, fil, conv2d_params);
  // Transform input (however many times it can be broken down by a 4x4 tile with stride of 2)
  TensorSizeStruct tile_size(std::vector<unsigned int>{1, 1, 4, 4});
  Conv2DParamPack conv2d_params_for_tiles(std::vector<unsigned int>{2, conv2d_params.padding(), 4});

  auto num_tiles_per_row = conv2d_params_for_tiles.get_out_sizes(inp.shape(), tile_size)[2];
  auto num_tiles_per_col = num_tiles_per_row;

  std::cout << "Number of tiles per row: " << num_tiles_per_row << ", number of tiles per col: " << num_tiles_per_col << "\n";

  std::cout << "Original tensor data (first in batch, channel 0): \n";
  inp.print(0);

  std::vector<std::vector<float>> transformed_inputs(num_tiles_per_row * num_tiles_per_col);
  for(int i = 0; i < 1; ++i) {
    for(int j = 0; j < 1; ++j) {
      transformed_inputs[i*num_tiles_per_row + j]
        = transform_input(inp.data(), fil.shape()[1], inp.shape()[2], inp.shape()[3], conv2d_params.padding(), i, j);
      for(int x = 0; x < 4; ++x) {
        for(int y = 0; y < 4; ++y) {
          std::cout << transformed_inputs[i*num_tiles_per_row + j][x*4 + y] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n";
    }
  }
  // Transform filter (1x)
  // ...
  // Batched matrix-multiply
  // ...
  // Inverse transform (same number of times as input, since we have that many results to invert)
  // ...
  // Done!
  return out;
}

int main() {
  Conv2DParamPack conv2dParams(std::vector<unsigned int>{1, 0, 3});
  TensorSizeStruct input_sizes{std::vector<unsigned int>{1, 1, 8, 8}};
  TensorSizeStruct filter_sizes{std::vector<unsigned int>{1, 1, 3, 3}};
  Tensor input(input_sizes), filter(filter_sizes);
  auto output = conv2d(input, filter, conv2dParams);
  auto out_shape = output.shape();
  for (auto& dimension : out_shape.get_dimensions())
    std::cout << dimension << " ";
  std::cout << "\n";
}
