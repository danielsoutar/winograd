import numpy as np

def conv2d_multi_channel(inp, w):
    """Two-dimensional convolution with multiple channels.

    Uses VALID padding, a stride of 1 and no dilation.

    input: input array with shape (in_channels, height, width) (CHW)
    w: filter array with shape (out_channels, in_channels, f_height, f_width) (FCHW)
       in_channels must match between input and filter.

    Returns a result with shape (out_channels, out_height, out_width).
    """
    batch, in_channels, in_height, in_width = inp.shape
    assert in_channels == w.shape[1]
    derive_out_dim = lambda dim, p, k, s: int(np.floor((dim + 2*p - k) / s)) + 1
    out_height = derive_out_dim(in_height, 0, w.shape[2], 1)
    out_width = derive_out_dim(in_width, 0, w.shape[3], 1)
    out_channels = w.shape[0]
    output = np.zeros((batch, out_channels, out_height, out_width))

    for b in range(batch):
        for out_c in range(out_channels):
            # For each output channel, perform 2d convolution summed across all
            # input channels.
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        for fi in range(w.shape[2]):
                            for fj in range(w.shape[3]):
                                w_element = w[out_c, c, fi, fj]
                                x_element = inp[b, c, i + fi, j + fj]
                                output[b, out_c, i, j] += x_element * w_element
    return output
