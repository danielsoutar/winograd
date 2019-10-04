import numpy as np

def conv2d_multi_channel(inp, w, padding=0):
    """2D convolution for NCHW inputs and FCHW filters.

    Uses a stride of 1 and no dilation. Assumes square inputs and filters.

    input: input with shape (batch_size, in_channels, height, width) (NCHW)
    w: filter with shape (out_channels, in_channels, f_height, f_width) (FCHW)
       in_channels must match between input and filter.

    Returns a result with shape (out_channels, out_height, out_width).
    """
    batch, in_channels, in_height, in_width = inp.shape
    out_channels, _, f_height, f_width = w.shape

    assert in_channels == w.shape[1]

    derive_out_dim = lambda dim, p, k, s: int(np.floor((dim + 2*p - k) / s)) + 1

    out_height = derive_out_dim(in_height, padding, f_height, 1)
    out_width = derive_out_dim(in_width, padding, f_width, 1)

    output = np.zeros((batch, out_channels, out_height, out_width))


    pfh = (f_height // 2) if padding > 0 else 0
    pfw = (f_width // 2) if padding > 0 else 0

    padded_input = np.pad(inp,  # ensure batch/channel dimensions are not padded
                          pad_width=((0, 0), (0, 0), (pfh, pfh), (pfw, pfw)),
                          mode='constant', constant_values=0)

    for b in range(batch):
        for out_c in range(out_channels):
            # For each output channel, perform 2d convolution summed across all
            # input channels.
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        for fi in range(f_height):
                            for fj in range(f_width):
                                w_element = w[out_c, c, fi, fj]
                                x_element = padded_input[b, c, i + fi, j + fj]
                                output[b, out_c, i, j] += x_element * w_element
    return output


def create_iota_tensor_from(d1, d2, d3, d4):
    return np.array([i for i in range(d1*d2*d3*d4)]).reshape((d1, d2, d3, d4))

if __name__ == "__main__":
    N, C, H, W = 1, 1, 8, 8
    F, Fh, Fw = 1, 3, 3
    inp = create_iota_tensor_from(N, C, H, W)
    fil = create_iota_tensor_from(F, C, Fh, Fw)

    out = conv2d_multi_channel(inp, fil, padding=1)

    print(out)
