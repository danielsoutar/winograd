// winograd_sequential.cpp - Implementation of the Winograd transform for
// convolutions.
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
std::vector<float> transform_input(std::vector<float> input,
                                   TensorSizeStruct inp_shape, int E,
                                   int padding, int tile_i, int tile_j) {
  auto N = inp_shape[0];
  auto C = inp_shape[1];
  auto H = inp_shape[2];
  auto W = inp_shape[3];

  int HW = H * W;

  std::vector<float> tile(E * N * C);

  int r = (0 - padding) + tile_i * 2;
  int c = (0 - padding) + tile_j * 2;

  int tile_size = 4;

  for (int b = 0; b < N; ++b) {
    for (int f = 0; f < C; ++f) {
      for (int input_i = r; input_i < r + tile_size; ++input_i) {
        for (int input_j = c; input_j < c + tile_size; ++input_j) {
          auto idx = (input_i - r) * tile_size * N * C + (input_j - c) * N * C +
                     b * C + f;
          if (input_i > -1 && input_i < H && input_j > -1 && input_j < W)
            tile[idx] = input[b * C * HW + f * HW + input_i * W + input_j];
          else
            tile[idx] = 0.0f;
        }
      }
    }
  }

  // std::cout << "Current slice of input at tile (" << tile_i << ", " << tile_j
  // << "):\n"; for(int ti = 0; ti < tile_size; ++ti) {
  //   for(int tj = 0; tj < tile_size; ++tj) {
  //     for(int b = 0; b < N; ++b) {
  //       for(int f = 0; f < C; ++f) {
  //         std::cout << std::setw(4) << tile[ti * tile_size * N * C + tj * N *
  //         C + b * C + f] << " ";
  //       }
  //       std::cout << "\n";
  //     }
  //     std::cout << "\n";
  //   }
  // }

  // Now have the slices of the input. For each channel, perform the input
  // transform. TIME TO DO ADDITION
#ifndef TILE
#define TILE(X) tile[X * N * C + b * C + f]
  for (int b = 0; b < N; ++b) {
    for (int f = 0; f < C; ++f) {
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
  }
#undef TILE
#endif

  return tile;
}

/**
 * Transform the filter for the Winograd algorithm. This means taking in a 3x3
 * filter to do convolutions with, and returning a C x E x F output, where C is
 * the number of input channels, F the number of output channels, and E the tile
 * size squared (16 for R, S = 3).
 *
 * In our case, we take the c-th 3x3 input feature map. This is done in
 * row-major order.
 */
std::vector<float> transform_filter(std::vector<float> filter, int R, int S,
                                    int C, int F, int tile_size) {
  int E = tile_size * tile_size;
  std::vector<float> tile(C * E * F);

  // Now have the slice of the filter. For each channel, perform the filter
  // transform. TIME TO DO ADDITION AND DIVISION
#ifndef TILE
#define TILE(X) tile[c * E * F + X * F + f]
#ifndef g
#define g(X) filter[f * C * R * S + c * R * S + X]
  for (int f = 0; f < F; ++f) {
    for (int c = 0; c < C; ++c) {
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
  }

  // std::cout << "Current feature-map:\n";
  // for (int f = 0; f < F; ++f) {
  //   for (int c = 0; c < C; ++c) {
  //     for (int i = 0; i < E; ++i) {
  //       std::cout << std::setw(5) << tile[f * C * E + c * E + i] << " ";
  //     }
  //     std::cout << "\n";
  //   }
  // }

  return tile;
}

std::vector<float> inverse_transform(std::vector<float> M, int E,
                                     TensorSizeStruct out_shape,
                                     int n_tiles_rows) {
  auto outN = out_shape[0];
  auto outC = out_shape[1];
  auto outH = out_shape[2];
  auto outW = out_shape[3];

  auto outHW = outH * outW;

  std::vector<float> Y(outN * outC * outH * outW);

  // A^T =  [ 1   1   1   0]
  //        [ 0   1  -1  -1]
  //
  // Let inverse transform = A^T * M * A
  int t_i = 0;
#ifndef M
#define M(I) M[t_i * E * outN * outC + I * outN * outC + b * outC + f]
  for (int b = 0; b < outN; ++b) {
    t_i = 0;
    for (int r = 0; r < outH; r += 2) {
      for (int c = 0; c < n_tiles_rows; ++c) {
        for (int f = 0; f < outC; ++f) {
          Y[b * outC * outHW + f * outHW + r * outW + c * 2] =
              M(0) + M(1) + M(2) + M(4) + M(5) + M(6) + M(8) + M(9) + M(10);
          Y[b * outC * outHW + f * outHW + r * outW + c * 2 + 1] =
              M(1) + M(5) + M(9) - M(2) - M(6) - M(10) - M(3) - M(7) - M(11);
          Y[b * outC * outHW + f * outHW + (r + 1) * outW + c * 2] =
              M(4) + M(5) + M(6) - M(8) - M(9) - M(10) - M(12) - M(13) - M(14);
          Y[b * outC * outHW + f * outHW + (r + 1) * outW + c * 2 + 1] =
              M(5) - M(9) - M(13) - M(6) + M(10) + M(14) - M(7) + M(11) + M(15);
        }
        ++t_i;
      }
    }
  }
#undef M
#endif

  std::cout << "t_i = " << t_i << "\n";

  return Y;
}

// Perform a 2D convolution using the Winograd transform
Tensor conv2d(Tensor inp, Tensor fil, Conv2DParamPack conv2d_params) {
  // Assume M and N are fixed, likewise R and S
  int m = 2, n = 2;
  int r = 3, s = 3;

  unsigned int A = m + r - 1;
  unsigned int B = n + s - 1;

  // Variables for the sizes of the inputs
  int batch_size = inp.shape()[0];
  int in_channels = inp.shape()[1];
  int in_height = inp.shape()[2];
  int in_width = inp.shape()[3];

  int out_channels = fil.shape()[0];
  int fil_channels = fil.shape()[1];
  int fil_height = fil.shape()[2];
  int fil_width = fil.shape()[3];

  // Initialise an output tensor based on inputs and operation.
  Tensor out(inp, fil, conv2d_params);
  // Transform input (however many times it can be broken down by a 4x4 tile
  // with stride of 2)
  TensorSizeStruct tile_size(std::vector<unsigned int>{1, 1, A, B});
  Conv2DParamPack conv2d_params_for_tiles(
      std::vector<unsigned int>{2, conv2d_params.padding(), A});

  auto n_tiles_rows =
      conv2d_params_for_tiles.get_out_sizes(inp.shape(), tile_size)[2];
  auto n_tiles_cols = n_tiles_rows;
  int T = n_tiles_rows * n_tiles_cols;

  // inp.print();

  // We want an input of shape [T, E, N, C], where E denotes the number of
  // elements in the intermediate tensor to do a pointwise multiply, and T the
  // total number of tiles. C denotes the number of input channels.
  std::vector<std::vector<float>> U(n_tiles_rows * n_tiles_cols);
  int E = A * B;
  for (int i = 0; i < n_tiles_rows; ++i)
    for (int j = 0; j < n_tiles_cols; ++j)
      U[i * n_tiles_cols + j] = transform_input(inp.data(), inp.shape(), E,
                                                conv2d_params.padding(), i, j);

  // Transform filter (1x)
  fil.print();

  std::vector<float> V =
      transform_filter(fil.data(), r, s, in_channels, out_channels, A);

  // Batched matrix-multiply per tile, per block of U and V
  std::vector<float> M(T * E * batch_size * out_channels);
  std::fill(M.begin(), M.end(), 0);
  for (int t = 0; t < T; ++t)
    for (int e = 0; e < E; ++e)
      for (int b = 0; b < batch_size; ++b)
        for (int c = 0; c < in_channels; ++c)
          for (int f = 0; f < out_channels; ++f)
            M[t * E * batch_size * out_channels +
              e * batch_size * out_channels + b * out_channels + f] +=
                U[t][e * batch_size * in_channels + b * in_channels + c] *
                V[c * E * out_channels + e * out_channels + f];

  // Inverse transform (same number of times as input, since we have that many
  // results to invert)
  auto out_shape = out.shape();
  std::vector<float> Y = inverse_transform(M, E, out.shape(), n_tiles_rows);

  auto outH = out_shape[2];
  auto outW = out_shape[3];
  auto outHW = outH * outW;

  out.set_data(Y);
  out.print();

  // Done!
  return out;
}

int main() {
  Conv2DParamPack conv2dParams(std::vector<unsigned int>{1, 0, 3});
  unsigned int N = 1, C = 1, H = 64, W = 64;
  unsigned int F = 2, Fh = 3, Fw = 3;
  TensorSizeStruct input_sizes{std::vector<unsigned int>{N, C, H, W}};
  TensorSizeStruct filter_sizes{std::vector<unsigned int>{F, C, Fh, Fw}};
  Tensor input(input_sizes), filter(filter_sizes);
  auto output = conv2d(input, filter, conv2dParams);
  auto out_shape = output.shape();
  std::cout << "Output dimensions for convolution: \n";
  for (auto &dimension : out_shape.get_dimensions())
    std::cout << dimension << " ";
  std::cout << "\n";
}
