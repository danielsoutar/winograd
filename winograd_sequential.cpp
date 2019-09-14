// winograd_sequential.cpp - Implementation of the Winograd transform for convolutions
// Created by: Daniel Soutar
#include "tensor.hpp"
#include <iostream>
#include <vector>

/**
 * The Winograd algorithm here performs a 2D convolution on a 4-dimensional
 * input. The input data is assumed to be in NCHW format, where:
 *
 *  > N denotes the minibatch size
 *  > C denotes the number of input channels or feature maps
 *  > H denotes the input's height
 *  > W denotes the input's width
 *
 * The filter is assumed to be in FCHW format, where:
 *
 *  > F denotes the number of output channels or feature maps
 *  > C denotes the number of input channels or feature maps (must match the
 * input) > H denotes the filter's height > W denotes the filter's width
 *
 * Dilation > 1 is not supported. This operation is only supported for stride=1
 * for performance reasons.
 */

/**
 * Transform the input for the Winograd algorithm. This means taking in a 4x4
 * slice of the input containing 4 3x3 slices to do convolutions on, and
 * returning a E x C output, where C is the number of input channels and E the
 * tile size squared (16 for R, S = 3).
 *
 * In our case, we take the (i,j) 4x4 slice. This is done in row-major order.
 */
std::vector<float> transform_input(std::vector<float> input, int channels,
                                   int height, int width, int padding,
                                   int tile_i, int tile_j) {
  std::vector<float> tile(4 * 4 * channels);
  int hw = height * width;
  int r = (0 - padding) + tile_i * 2;
  int c = (0 - padding) + tile_j * 2;
  int tile_size = 4;
  for (int channel = 0; channel < channels; ++channel) {
    int channel_offset = channel * hw;
    int channel_tile_offset = channel;
    for (int input_i = r; input_i < r + tile_size; ++input_i) {
      for (int input_j = c; input_j < c + tile_size; ++input_j) {
        auto idx = channel_tile_offset + (input_i - r) * tile_size * channels +
                   (input_j - c) * channels;
        if (input_i > -1 && input_i < height && input_j > -1 && input_j < width)
          tile[idx] = input[channel_offset + input_i * width + input_j];
        else
          tile[idx] = 0.0f;
      }
    }
  }

  std::cout << "Current slice of input:\n";
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < tile_size; ++i) {
      for (int j = 0; j < tile_size; ++j) {
        std::cout << std::setw(4)
                  << tile[c + i * tile_size * channels + j * channels] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  // Now have the slices of the input. For each channel, perform the input
  // transform. TIME TO DO ADDITION
#ifndef TILE
#define TILE(X) tile[X * channels + channel]
  for (int channel = 0; channel < channels; ++channel) {
    float bd0 = TILE(0) - TILE(8);
    float bd1 = TILE(1) - TILE(9);
    float bd2 = TILE(2) - TILE(10);
    float bd3 = TILE(3) - TILE(11);
    float bd4 = TILE(4) + TILE(8);
    float bd5 = TILE(5) + TILE(9);
    float bd6 = TILE(6) + TILE(10);
    float bd7 = TILE(7) + TILE(11);
    float bd8 = TILE(8) - TILE(4);
    float bd9 = TILE(9) - TILE(5);
    float bd10 = TILE(10) - TILE(6);
    float bd11 = TILE(11) - TILE(7);
    float bd12 = TILE(4) - TILE(12);
    float bd13 = TILE(5) - TILE(13);
    float bd14 = TILE(6) - TILE(14);
    float bd15 = TILE(7) - TILE(15);

    TILE(0) = bd0 - bd2;
    TILE(1) = bd1 + bd2;
    TILE(2) = bd2 - bd1;
    TILE(3) = bd1 - bd3;

    TILE(4) = bd4 - bd6;
    TILE(5) = bd5 + bd6;
    TILE(6) = bd6 - bd5;
    TILE(7) = bd5 - bd7;

    TILE(8) = bd8 - bd10;
    TILE(9) = bd9 + bd10;
    TILE(10) = bd10 - bd9;
    TILE(11) = bd9 - bd11;

    TILE(12) = bd12 - bd14;
    TILE(13) = bd13 + bd14;
    TILE(14) = bd14 - bd13;
    TILE(15) = bd13 - bd15;
  }
#undef TILE
#endif

  return tile;
}

/**
 * Transform the filter for the Winograd algorithm. This means taking in a 3x3
 * filter to do convolutions with, and returning a C x E output, where C is the
 * number of input channels and E the tile size squared (16 for R, S = 3).
 *
 * In our case, we take the c-th 3x3 input feature map. This is done in
 * row-major order.
 */
std::vector<float> transform_filter(std::vector<float> filter, int R, int S,
                                    int channels, int tile_size) {
  std::vector<float> tile(tile_size * tile_size * channels);

  // Now have the slice of the filter. For each channel, perform the filter
  // transform. TIME TO DO ADDITION AND DIVISION
#ifndef TILE
#define TILE(X) tile[c * tile_size * tile_size + X]
#ifndef g
#define g(X) filter[c * R * S + X]
  for (int c = 0; c < channels; ++c) {
    float g0_g2 = (g(0) + g(2));
    float g0_g6 = (g(0) + g(6));
    float g0_g3_g6 = (g0_g6 + g(3));
    float g2_g8 = (g(2) + g(8));
    float g2_g5_g8 = (g2_g8 + g(5));
    float g1_g7 = (g(1) + g(7));
    float g1_g4_g7 = (g1_g7 + g(4));
    float g0_g6_sub_g3 = (g0_g6 - g(3));
    float g1_g7_sub_g4 = (g1_g7 - g(4));
    float g2_g8_sub_g5 = g2_g8 - g(5);
    float g6_g8 = g(6) + g(8);
    float g6_g7_g8 = (g6_g8 + g(7));
    TILE(0) = g(0);
    TILE(1) = (g0_g2 + g(1)) / 2;
    TILE(2) = (g0_g2 - g(1)) / 2;
    TILE(3) = g(2);
    TILE(4) = g0_g3_g6 / 2;
    TILE(5) = (g0_g3_g6 + g2_g5_g8 + g1_g4_g7) / 4;
    TILE(6) = (g0_g3_g6 - g1_g4_g7 + g2_g5_g8) / 4;
    TILE(7) = g2_g5_g8 / 2;
    TILE(8) = g0_g6_sub_g3 / 2;
    TILE(9) = (g0_g6_sub_g3 + g1_g7_sub_g4 + g2_g8_sub_g5) / 4;
    TILE(10) = (g0_g6_sub_g3 - g1_g7_sub_g4 + g2_g8_sub_g5) / 4;
    TILE(11) = g2_g8_sub_g5 / 2;
    TILE(12) = g(6);
    TILE(13) = g6_g7_g8 / 2;
    TILE(14) = (g6_g8 - g(7)) / 2;
    TILE(15) = g(8);
#undef g
#undef TILE
#endif
#endif
  }

  std::cout << "Current feature-map:\n";
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < tile_size * tile_size; ++i) {
      std::cout << std::setw(5) << tile[c * tile_size * tile_size + i] << " ";
    }
    std::cout << "\n";
  }

  return tile;
}

std::vector<float> inverse_transform(std::vector<float> M, int out_rows,
                                     int out_cols, int n_tiles_rows) {
  std::vector<float> Y(out_rows * out_cols);
  // A^T =  [ 1   1   1   0]
  //        [ 0   1  -1  -1]
  //
  // Let inverse transform = A^T * M * A

  int t_i = 0;
#ifndef M
#define M(I) M[t_i * 16 + I]
  for (int r = 0; r < out_rows; r += 2) {
    for (int i = 0; i < n_tiles_rows; ++i) {
      Y[r * out_cols + i * 2] =
          M(0) + M(1) + M(2) + M(4) + M(5) + M(6) + M(8) + M(9) + M(10);
      Y[r * out_cols + i * 2 + 1] =
          M(1) + M(5) + M(9) - M(2) - M(6) - M(10) - M(3) - M(7) - M(11);
      Y[(r + 1) * out_cols + i * 2] =
          M(4) + M(5) + M(6) - M(8) - M(9) - M(10) - M(12) - M(13) - M(14);
      Y[(r + 1) * out_cols + i * 2 + 1] =
          M(5) - M(9) - M(13) - M(6) + M(10) + M(14) - M(7) + M(11) + M(15);

      ++t_i;
    }
  }
#undef M
#endif

  return Y;
}

// Perform a 2D convolution using the Winograd transform
Tensor conv2d(Tensor inp, Tensor fil, Conv2DParamPack conv2d_params) {
  // Initialise an output tensor based on inputs and operation.
  Tensor out(inp, fil, conv2d_params);
  // Transform input (however many times it can be broken down by a 4x4 tile
  // with stride of 2)
  TensorSizeStruct tile_size(std::vector<unsigned int>{1, 1, 4, 4});
  Conv2DParamPack conv2d_params_for_tiles(
      std::vector<unsigned int>{2, conv2d_params.padding(), 4});

  auto n_tiles_rows =
      conv2d_params_for_tiles.get_out_sizes(inp.shape(), tile_size)[2];
  auto n_tiles_cols = n_tiles_rows;
  int T = n_tiles_rows * n_tiles_cols;

  std::cout << "Number of tiles per row: " << n_tiles_rows
            << ", number of tiles per col: " << n_tiles_cols << "\n";

  std::cout << "Original tensor data (first in batch, channel 0): \n";
  inp.print();
  // inp.print(0);
  // std::cout << "Original tensor data (first in batch, channel 1): \n";
  // inp.print(1);

  // We want an input of shape [T, E, N, C], where E denotes the number of
  // elements in the intermediate tensor to do a pointwise multiply, and T the
  // total number of tiles.
  std::vector<std::vector<float>> transformed_inputs(n_tiles_rows *
                                                     n_tiles_cols);
  for (int i = 0; i < n_tiles_rows; ++i) {
    for (int j = 0; j < n_tiles_cols; ++j) {
      transformed_inputs[i * n_tiles_cols + j] =
          transform_input(inp.data(), fil.shape()[1], inp.shape()[2],
                          inp.shape()[3], conv2d_params.padding(), i, j);
    }
  }
  // std::cout << "Printing transformed inputs\n";
  // for(int t = 0; t < T; ++t) {
  //   for(int x = 0; x < 4; ++x) {
  //     for(int y = 0; y < 4; ++y) {
  //       std::cout << transformed_inputs[t][x*4 + y] << " ";
  //     }
  //     std::cout << "\n";
  //   }
  //   std::cout << "\n\n";
  // }

  // Transform filter (1x)
  int C = inp.shape()[1];
  fil.print();
  std::vector<float> transformed_filter =
      transform_filter(fil.data(), 3, 3, C, 4);

  // Batched matrix-multiply
  int E = 4 * 4;
  std::vector<float> M(T * E);
  std::fill(M.begin(), M.end(), 0);
  for (int i = 0; i < T; ++i)
    for (int j = 0; j < E; ++j)
      for (int c = 0; c < C; ++c)
        M[i * E + j] +=
            transformed_inputs[i][j * C + c] * transformed_filter[c * E + j];

  // Inverse transform (same number of times as input, since we have that many
  // results to invert)
  auto out_shape = out.shape();
  std::vector<float> Y =
      inverse_transform(M, out_shape[2], out_shape[3], n_tiles_rows);

  for (int i = 0; i < out_shape[2]; ++i) {
    for (int j = 0; j < out_shape[3]; ++j) {
      std::cout << std::setw(5) << Y[i * out_shape[3] + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  out.set_data(Y);
  // out.print();
  // Done!
  return out;
}

int main() {
  Conv2DParamPack conv2dParams(std::vector<unsigned int>{1, 0, 3});
  TensorSizeStruct input_sizes{std::vector<unsigned int>{1, 2, 8, 8}};
  TensorSizeStruct filter_sizes{std::vector<unsigned int>{1, 2, 3, 3}};
  Tensor input(input_sizes), filter(filter_sizes);
  auto output = conv2d(input, filter, conv2dParams);
  auto out_shape = output.shape();
  std::cout << "Output dimensions for convolution: \n";
  for (auto &dimension : out_shape.get_dimensions())
    std::cout << dimension << " ";
  std::cout << "\n";
}
