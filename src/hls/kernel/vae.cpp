#include <stdio.h>
#include <string.h>
#include <ap_fixed.h>

#define KERNEL_SIZE 2
#define STRIDE 2
#define INPUT_SIZE 64
#define OUTPUT_SIZE_1 ((INPUT_SIZE - KERNEL_SIZE) / STRIDE + 1) //32
#define OUTPUT_SIZE_2 ((OUTPUT_SIZE_1 - KERNEL_SIZE) / STRIDE + 1) //16
#define OUTPUT_SIZE_3 ((OUTPUT_SIZE_2 - KERNEL_SIZE) / STRIDE + 1) //8
#define OUTPUT_SIZE_4 ((OUTPUT_SIZE_3 - KERNEL_SIZE) / STRIDE + 1) //4
#define INPUT_CHANNELS 3
#define NUM_FILTERS_1 16
#define NUM_FILTERS_2 32
#define NUM_FILTERS_3 64
#define NUM_FILTERS_4 128
#define FC_INPUT_SIZE 2048  // (4 * 4 * 128)
#define FC_OUTPUT_SIZE 32   // z_mean & z_log_var
#define DEC_FC_OUTPUT   2048
#define DEC_FC_INPUT 32
#define DEC_NUM_FILTERS_1 64  // Conv2DTranspose (None, 8, 8, 64)
#define DEC_NUM_FILTERS_2 32  // Conv2DTranspose (None, 16, 16, 32)
#define DEC_NUM_FILTERS_3 16  // Conv2DTranspose (None, 32, 32, 16)
#define DEC_NUM_FILTERS_4 3   // Conv2DTranspose (None, 64, 64, 3)
#define DEC_TRANSPOSE_1 8
#define DEC_TRANSPOSE_2 16
#define DEC_TRANSPOSE_3 32
#define DEC_TRANSPOSE_4 64
typedef ap_fixed<16, 4, AP_RND_CONV, AP_SAT>  fixed_t;
#define INDEX_2D(i, j, width) ((i) * (width) + (j))
#define INDEX_3D(c, i, j, height, width) ((c) * (height) * (width) + (i) * (width) + (j))
#define INDEX_4D(f, c, h, w, C, H, W) ((f) * (C) * (H) * (W) + (c) * (H) * (W) + (h) * (W) + (w))
void relu(fixed_t *data, int size) {
    LOOP_RELU:
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}

void convolution_multifilter(
    fixed_t *input,
    fixed_t *kernel,
    fixed_t *output,
    int input_size,
    int kernel_size,
    int output_size,
    int stride,
    int input_channels,
    int num_filters,
    fixed_t *bias) {

    LOOP_FILTER:
    for (int f = 0; f < num_filters; f++) {
        LOOP_OUTPUT_I:
        for (int i = 0; i < output_size; i++) {
            LOOP_OUTPUT_J:
            for (int j = 0; j < output_size; j++) {
                fixed_t sum = bias[f];

                LOOP_CHANNEL:
                for (int c = 0; c < input_channels; c++) {
                    LOOP_KERNEL_I:
                    for (int ki = 0; ki < kernel_size; ki++) {
                        LOOP_KERNEL_J:
                        for (int kj = 0; kj < kernel_size; kj++) {

                            int input_idx = INDEX_3D(c, i * stride + ki, j * stride + kj, input_size, input_size);
                            int kernel_idx = INDEX_4D(f, c, ki, kj, input_channels, kernel_size, kernel_size);
                            sum += input[input_idx] * kernel[kernel_idx];
                        }
                    }
                }

                output[INDEX_3D(f, i, j, output_size, output_size)] = sum;
            }
        }
    }

    relu(output, num_filters * output_size * output_size);
}

void fully_connected(
    fixed_t *input,
    fixed_t *weights,
    fixed_t *bias,
    fixed_t *output,
    int input_size,
    int output_size,
    int mode) {

    LOOP_FC_OUT:
    for (int i = 0; i < output_size; i++) {
        fixed_t sum = bias[i];

        LOOP_FC_IN:
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[i * input_size + j];
        }

        output[i] = sum;
    }

    if (mode == 1) {
        relu(output, output_size);
    }
}

void flatten(fixed_t *input, fixed_t *output, int channels, int height, int width) {
    int index = 0;

    LOOP_FLATTEN_C:
    for (int c = 0; c < channels; c++) {
        LOOP_FLATTEN_H:
        for (int i = 0; i < height; i++) {
            LOOP_FLATTEN_W:
            for (int j = 0; j < width; j++) {
                output[index++] = input[INDEX_3D(c, i, j, height, width)];
            }
        }
    }
}

void compute_epsilon(fixed_t *z_log_var, fixed_t *epsilon, int size) {
    LOOP_EPSILON:
    for (int i = 0; i < size; i++) {
        fixed_t x = fixed_t(0.5) * z_log_var[i];
        fixed_t x2 = x * x;
        fixed_t x3 = x2 * x;
        fixed_t x4 = x3 * x;

        epsilon[i] = fixed_t(1.0) + x + (x2 * fixed_t(0.5)) + (x3 * fixed_t(0.1667)) + (x4 * fixed_t(0.04167));
    }
}
void compute_z_sampling(fixed_t *fc_output_buffer_1,
                        fixed_t *fc_output_buffer_2,
                        fixed_t *epsilon_buffer,
                        fixed_t *z_buffer, int size) {
    LOOP_Z_SAMPLING: for (int i = 0; i < size; i++) {
        z_buffer[i] = fc_output_buffer_1[i] + fc_output_buffer_2[i] * epsilon_buffer[i];
    }
}

void sigmoid(fixed_t *data, int size) {
    loop_i: for (int i = 0; i < size; i++) {
        fixed_t x = data[i];

        // Nếu x >= 4.0, sigmoid(x) = 1
        if (x >= fixed_t(4.0)) {
            data[i] = fixed_t(1.0);
        }
        // Nếu 0 <= x < 4.0, áp dụng công thức xấp xỉ bậc hai
        else if (x >= fixed_t(0.0)) {
            data[i] = fixed_t(-0.03125) * x * x + fixed_t(0.25) * x + fixed_t(0.5);
        }
        // Nếu x < 0, áp dụng tính chất đối xứng: sigmoid(x) = 1 - sigmoid(-x)
        else {
            fixed_t pos_x = -x;
            if (pos_x >= fixed_t(4.0)) {
                data[i] = fixed_t(0.0);
            } else {
                data[i] = fixed_t(1.0) - (fixed_t(-0.03125) * pos_x * pos_x + fixed_t(0.25) * pos_x + fixed_t(0.5));
            }
        }
    }
}

void conv2d_transpose(
    fixed_t* input, fixed_t* kernel, fixed_t* output, fixed_t* bias,
    int input_h, int input_w, int input_channels,
    int output_h, int output_w, int num_filters,
    int kernel_size, int stride, int mode)
{
    LOOP_INIT_OUTPUT:
    for (int i = 0; i < num_filters * output_h * output_w; i++) {
        output[i] = 0;
    }

    LOOP_TRANS_FILTER:
    for (int f = 0; f < num_filters; f++) {
        LOOP_TRANS_H:
        for (int h = 0; h < input_h; h++) {
            LOOP_TRANS_W:
            for (int w = 0; w < input_w; w++) {
                LOOP_TRANS_C:
                for (int c_in = 0; c_in < input_channels; c_in++) {
                    fixed_t input_val = input[INDEX_3D(c_in, h, w, input_h, input_w)];

                    LOOP_TRANS_KH:
                    for (int kh = 0; kh < kernel_size; kh++) {
                        LOOP_TRANS_KW:
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int out_h = h * stride + kh;
                            int out_w = w * stride + kw;

                            if (out_h < output_h && out_w < output_w) {
                                output[INDEX_3D(f, out_h, out_w, output_h, output_w)] +=
                                    input_val * kernel[INDEX_4D(f, c_in, kh, kw, input_channels, kernel_size, kernel_size)];
                            }
                        }
                    }
                }
            }
        }
    }

    LOOP_BIAS_F:
    for (int f = 0; f < num_filters; f++) {
        LOOP_BIAS_H:
        for (int h = 0; h < output_h; h++) {
            LOOP_BIAS_W:
            for (int w = 0; w < output_w; w++) {
                output[INDEX_3D(f, h, w, output_h, output_w)] += bias[f];
            }
        }
    }

    if (mode == 1) {
        relu(output, num_filters * output_h * output_w);
    } else if (mode == 2) {
        sigmoid(output, num_filters * output_h * output_w);
    }
}
void reshape(const fixed_t *input, fixed_t *output, int C, int H, int W) {
    loop_c: for (int c = 0; c < C; ++c) {
        loop_h: for (int h = 0; h < H; ++h) {
            loop_w: for (int w = 0; w < W; ++w) {
                output[c * (H * W) + h * W + w] = input[c + w * C + h * (C * W)];
            }
        }
    }
}
void memcpy_hls(fixed_t *dest, const fixed_t *src, int size) {
	LOOP_MEMCPY:
	for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

void vae_model(
    volatile fixed_t *input,
	//Encoder
    fixed_t kernel_1[NUM_FILTERS_1][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    fixed_t kernel_2[NUM_FILTERS_2][NUM_FILTERS_1][KERNEL_SIZE][KERNEL_SIZE],
    fixed_t kernel_3[NUM_FILTERS_3][NUM_FILTERS_2][KERNEL_SIZE][KERNEL_SIZE],
    fixed_t kernel_4[NUM_FILTERS_4][NUM_FILTERS_3][KERNEL_SIZE][KERNEL_SIZE],
    fixed_t bias_1[NUM_FILTERS_1],
    fixed_t bias_2[NUM_FILTERS_2],
    fixed_t bias_3[NUM_FILTERS_3],
    fixed_t bias_4[NUM_FILTERS_4],
	//2 lop FC
    fixed_t fc_weight_1[FC_OUTPUT_SIZE][FC_INPUT_SIZE],
    fixed_t fc_weight_2[FC_OUTPUT_SIZE][FC_INPUT_SIZE],
    fixed_t fc_bias_1[FC_OUTPUT_SIZE],
    fixed_t fc_bias_2[FC_OUTPUT_SIZE],
	//Dense Decoder
	fixed_t dec_dense_weight[DEC_FC_OUTPUT][DEC_FC_INPUT],
	fixed_t dec_dense_bias[DEC_FC_OUTPUT],
	//Decoder
	fixed_t dec_kernel_1[DEC_NUM_FILTERS_1][NUM_FILTERS_4][KERNEL_SIZE][KERNEL_SIZE],
	fixed_t dec_bias_1[DEC_NUM_FILTERS_1],
	fixed_t dec_kernel_2[DEC_NUM_FILTERS_2][DEC_NUM_FILTERS_1][KERNEL_SIZE][KERNEL_SIZE],
	fixed_t dec_bias_2[DEC_NUM_FILTERS_2],
	fixed_t dec_kernel_3[DEC_NUM_FILTERS_3][DEC_NUM_FILTERS_2][KERNEL_SIZE][KERNEL_SIZE],
	fixed_t dec_bias_3[DEC_NUM_FILTERS_3],
	fixed_t dec_kernel_4[INPUT_CHANNELS][DEC_NUM_FILTERS_3][KERNEL_SIZE][KERNEL_SIZE],
	fixed_t dec_bias_4[INPUT_CHANNELS],
    volatile fixed_t *epsilon_input,
    volatile fixed_t *z_output) {

    fixed_t input_buffer[INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE];
    fixed_t kernel_buffer_1[NUM_FILTERS_1 * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE];
    fixed_t kernel_buffer_2[NUM_FILTERS_2 * NUM_FILTERS_1 * KERNEL_SIZE * KERNEL_SIZE];
    fixed_t kernel_buffer_3[NUM_FILTERS_3 * NUM_FILTERS_2 * KERNEL_SIZE * KERNEL_SIZE];
    fixed_t kernel_buffer_4[NUM_FILTERS_4 * NUM_FILTERS_3 * KERNEL_SIZE * KERNEL_SIZE];

    fixed_t bias_buffer_1[NUM_FILTERS_1];
    fixed_t bias_buffer_2[NUM_FILTERS_2];
    fixed_t bias_buffer_3[NUM_FILTERS_3];
    fixed_t bias_buffer_4[NUM_FILTERS_4];

    fixed_t output_buffer_3[NUM_FILTERS_1 * OUTPUT_SIZE_1 * OUTPUT_SIZE_1];
    fixed_t output_buffer_4[NUM_FILTERS_1 * OUTPUT_SIZE_1 * OUTPUT_SIZE_1];

    fixed_t fc_weight_buffer_1[FC_OUTPUT_SIZE * FC_INPUT_SIZE];
    fixed_t fc_weight_buffer_2[FC_OUTPUT_SIZE * FC_INPUT_SIZE];
    fixed_t fc_bias_buffer_1[FC_OUTPUT_SIZE];
    fixed_t fc_bias_buffer_2[FC_OUTPUT_SIZE];

    fixed_t fc_output_buffer_1[FC_OUTPUT_SIZE];
    fixed_t fc_output_buffer_2[FC_OUTPUT_SIZE];
    fixed_t epsilon_buffer[FC_OUTPUT_SIZE];

    fixed_t dense_weight_buffer[FC_OUTPUT_SIZE * FC_INPUT_SIZE];
    fixed_t dense_bias_buffer[DEC_FC_OUTPUT];
    fixed_t dec_kernel_buffer_1[DEC_NUM_FILTERS_1 * NUM_FILTERS_4 * KERNEL_SIZE * KERNEL_SIZE];
    fixed_t dec_kernel_buffer_2[DEC_NUM_FILTERS_2 * DEC_NUM_FILTERS_1 * KERNEL_SIZE * KERNEL_SIZE];
    fixed_t dec_kernel_buffer_3[DEC_NUM_FILTERS_3 * DEC_NUM_FILTERS_2 * KERNEL_SIZE * KERNEL_SIZE];
    fixed_t dec_kernel_buffer_4[DEC_NUM_FILTERS_4 * DEC_NUM_FILTERS_3 * KERNEL_SIZE * KERNEL_SIZE];

    fixed_t dec_bias_buffer_1[DEC_NUM_FILTERS_1];
    fixed_t dec_bias_buffer_2[DEC_NUM_FILTERS_2];
    fixed_t dec_bias_buffer_3[DEC_NUM_FILTERS_3];
    fixed_t dec_bias_buffer_4[DEC_NUM_FILTERS_4];


	#pragma HLS INTERFACE m_axi port=input depth=64*64*3 offset=slave
	#pragma HLS INTERFACE m_axi port=kernel_1 depth=3*3*3*16 offset=slave
	#pragma HLS INTERFACE m_axi port=kernel_2 depth=3*3*16*32 offset=slave
	#pragma HLS INTERFACE m_axi port=kernel_3 depth=3*3*32*64 offset=slave
	#pragma HLS INTERFACE m_axi port=kernel_4 depth=3*3*64*128 offset=slave

	#pragma HLS INTERFACE m_axi port=bias_1 depth=16 offset=slave
	#pragma HLS INTERFACE m_axi port=bias_2 depth=32 offset=slave
	#pragma HLS INTERFACE m_axi port=bias_3 depth=64 offset=slave
	#pragma HLS INTERFACE m_axi port=bias_4 depth=128 offset=slave

	#pragma HLS INTERFACE m_axi port=fc_weight_1 depth=2048*32 offset=slave
	#pragma HLS INTERFACE m_axi port=fc_weight_2 depth=2048*32 offset=slave

	#pragma HLS INTERFACE m_axi port=fc_bias_1 depth=32 offset=slave
	#pragma HLS INTERFACE m_axi port=fc_bias_2 depth=32 offset=slave

	#pragma HLS INTERFACE m_axi port=dec_dense_weight depth=2048*32 offset=slave
	#pragma HLS INTERFACE m_axi port=dec_dense_bias depth=2048 offset=slave

    #pragma HLS INTERFACE m_axi port=epsilon_input depth=32 offset=slave
    #pragma HLS INTERFACE m_axi port=z_output depth=12288 offset=slave

	#pragma HLS INTERFACE m_axi port=dec_kernel_1 depth=3*3*128*64 offset=slave
	#pragma HLS INTERFACE m_axi port=dec_bias_1 depth=64 offset=slave

	#pragma HLS INTERFACE m_axi port=dec_kernel_2 depth=3*3*64*32 offset=slave
	#pragma HLS INTERFACE m_axi port=dec_bias_2 depth=32 offset=slave

	#pragma HLS INTERFACE m_axi port=dec_kernel_3 depth=3*3*32*16 offset=slave
	#pragma HLS INTERFACE m_axi port=dec_bias_3 depth=16 offset=slave

	#pragma HLS INTERFACE m_axi port=dec_kernel_4 depth=3*3*16*3 offset=slave
	#pragma HLS INTERFACE m_axi port=dec_bias_4 depth=3 offset=slave
	#pragma HLS INTERFACE s_axilite port=return

    memcpy_hls(input_buffer, (const fixed_t *)input, INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE);
    memcpy_hls(kernel_buffer_1, (const fixed_t *)kernel_1, NUM_FILTERS_1 * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE);
    memcpy_hls(kernel_buffer_2, (const fixed_t *)kernel_2, NUM_FILTERS_2 * NUM_FILTERS_1 * KERNEL_SIZE * KERNEL_SIZE);
    memcpy_hls(kernel_buffer_3, (const fixed_t *)kernel_3, NUM_FILTERS_3 * NUM_FILTERS_2 * KERNEL_SIZE * KERNEL_SIZE);
    memcpy_hls(kernel_buffer_4, (const fixed_t *)kernel_4, NUM_FILTERS_4 * NUM_FILTERS_3 * KERNEL_SIZE * KERNEL_SIZE);
    memcpy_hls(bias_buffer_1, bias_1,NUM_FILTERS_1);
    memcpy_hls(bias_buffer_2, bias_2,NUM_FILTERS_2);
    memcpy_hls(bias_buffer_3, bias_3, NUM_FILTERS_3);
    memcpy_hls(bias_buffer_4, bias_4,NUM_FILTERS_4);
    memcpy_hls(fc_weight_buffer_1, (const fixed_t *)fc_weight_1, FC_OUTPUT_SIZE * FC_INPUT_SIZE);
    memcpy_hls(fc_bias_buffer_1, (const fixed_t *)fc_bias_1, FC_OUTPUT_SIZE);
    memcpy_hls(fc_weight_buffer_1, (const fixed_t *)fc_weight_1, FC_OUTPUT_SIZE);
    memcpy_hls(epsilon_buffer, (const fixed_t *)epsilon_input, FC_OUTPUT_SIZE);
    memcpy_hls(dense_bias_buffer, (const fixed_t *)dec_dense_bias, DEC_FC_OUTPUT);
    convolution_multifilter(
        input_buffer,
        kernel_buffer_1,
        output_buffer_3,
        INPUT_SIZE,
        KERNEL_SIZE,
        OUTPUT_SIZE_1,
        STRIDE,
        INPUT_CHANNELS,
        NUM_FILTERS_1,
        bias_buffer_1);

    convolution_multifilter(
        output_buffer_3,
        kernel_buffer_2,
        output_buffer_4,
        OUTPUT_SIZE_1,
        KERNEL_SIZE,
        OUTPUT_SIZE_2,
        STRIDE,
        NUM_FILTERS_1,
        NUM_FILTERS_2,
        bias_buffer_2);

    convolution_multifilter(
        output_buffer_4,
        kernel_buffer_3,
        output_buffer_3,
        OUTPUT_SIZE_2,
        KERNEL_SIZE,
        OUTPUT_SIZE_3,
        STRIDE,
        NUM_FILTERS_2,
        NUM_FILTERS_3,
        bias_buffer_3);

    convolution_multifilter(
        output_buffer_3,
        kernel_buffer_4,
        output_buffer_4,
        OUTPUT_SIZE_3,
        KERNEL_SIZE,
        OUTPUT_SIZE_4,
        STRIDE,
        NUM_FILTERS_3,
        NUM_FILTERS_4,
        bias_buffer_4);



    fully_connected(
    		output_buffer_4, fc_weight_buffer_1, fc_bias_buffer_1, fc_output_buffer_1, FC_INPUT_SIZE, FC_OUTPUT_SIZE, 0);

    memcpy_hls(fc_weight_buffer_1, (const fixed_t *)fc_weight_2, FC_OUTPUT_SIZE * FC_INPUT_SIZE);

    fully_connected(
    		output_buffer_4, fc_weight_buffer_1, fc_bias_buffer_2, fc_output_buffer_2, FC_INPUT_SIZE, FC_OUTPUT_SIZE, 0);

    fixed_t z_buffer[FC_OUTPUT_SIZE];
    compute_epsilon(fc_output_buffer_2, fc_output_buffer_2, FC_OUTPUT_SIZE);

    compute_z_sampling(fc_output_buffer_1, fc_output_buffer_2, epsilon_buffer, z_buffer, FC_OUTPUT_SIZE);


    fixed_t dense_out[DEC_FC_OUTPUT];

    memcpy_hls(fc_weight_buffer_1, (const fixed_t *)dec_dense_weight, FC_OUTPUT_SIZE * FC_INPUT_SIZE);
    fully_connected(
    		z_buffer, fc_weight_buffer_1, dense_bias_buffer, output_buffer_4, DEC_FC_INPUT, DEC_FC_OUTPUT, 1);
    reshape(output_buffer_4, dense_out, 128, 4, 4);


    memcpy_hls(kernel_buffer_4, (const fixed_t *)dec_kernel_1, DEC_NUM_FILTERS_1 * NUM_FILTERS_4 * KERNEL_SIZE * KERNEL_SIZE);
    memcpy_hls(bias_buffer_3, (const fixed_t *)dec_bias_1, DEC_NUM_FILTERS_1);

    memcpy_hls(kernel_buffer_3, (const fixed_t *)dec_kernel_2, DEC_NUM_FILTERS_2 * DEC_NUM_FILTERS_1 * KERNEL_SIZE * KERNEL_SIZE);
    memcpy_hls(bias_buffer_2, (const fixed_t *)dec_bias_2, DEC_NUM_FILTERS_2);

    memcpy_hls(kernel_buffer_2, (const fixed_t *)dec_kernel_3, DEC_NUM_FILTERS_3 * DEC_NUM_FILTERS_2 * KERNEL_SIZE * KERNEL_SIZE);
    memcpy_hls(bias_buffer_1, (const fixed_t *)dec_bias_3, DEC_NUM_FILTERS_3);

    memcpy_hls(kernel_buffer_1, (const fixed_t *)dec_kernel_4, INPUT_CHANNELS * DEC_NUM_FILTERS_3 * KERNEL_SIZE * KERNEL_SIZE);
    memcpy_hls(dec_bias_buffer_4, (const fixed_t *)dec_bias_4, INPUT_CHANNELS);

    conv2d_transpose(
    	dense_out,
		kernel_buffer_4,
		output_buffer_3,
		bias_buffer_3,
		OUTPUT_SIZE_4, //input size 4
		OUTPUT_SIZE_4,
		NUM_FILTERS_4, //channels in
		DEC_TRANSPOSE_1, //output size 8
		DEC_TRANSPOSE_1,
		DEC_NUM_FILTERS_1,//channels out
        KERNEL_SIZE,
		STRIDE,
		1);
    conv2d_transpose(
		output_buffer_3,
		kernel_buffer_3,
		output_buffer_4,
		bias_buffer_2,
		DEC_TRANSPOSE_1, //input size 8
		DEC_TRANSPOSE_1,
		DEC_NUM_FILTERS_1, //channels in
		DEC_TRANSPOSE_2, //output size 16
		DEC_TRANSPOSE_2,
		DEC_NUM_FILTERS_2,//channels out
        KERNEL_SIZE,
		STRIDE,
		1);
    conv2d_transpose(
		output_buffer_4,
		kernel_buffer_2,
		output_buffer_3,
		bias_buffer_1,
		DEC_TRANSPOSE_2, //input size 16
		DEC_TRANSPOSE_2,
		DEC_NUM_FILTERS_2, //channels in
		DEC_TRANSPOSE_3, //output size 32
		DEC_TRANSPOSE_3,
		DEC_NUM_FILTERS_3,//channels out
        KERNEL_SIZE,
		STRIDE,
		1);
    conv2d_transpose(
		output_buffer_3,
		kernel_buffer_1,
		output_buffer_4,
		dec_bias_buffer_4,
		DEC_TRANSPOSE_3, //input size 32
		DEC_TRANSPOSE_3,
		DEC_NUM_FILTERS_3, //channels in
		DEC_TRANSPOSE_4, //output size 64
		DEC_TRANSPOSE_4,
		DEC_NUM_FILTERS_4,//channels out
        KERNEL_SIZE,
		STRIDE,
		2);

    memcpy_hls((fixed_t *)z_output, (const fixed_t *)output_buffer_4, DEC_TRANSPOSE_4 * DEC_TRANSPOSE_4 * DEC_NUM_FILTERS_4);

}
