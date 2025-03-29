#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ap_fixed.h>

#define KERNEL_SIZE 2
#define STRIDE 2
#define INPUT_SIZE 64
#define OUTPUT_SIZE_1 ((INPUT_SIZE - KERNEL_SIZE) / STRIDE + 1)
#define OUTPUT_SIZE_2 ((OUTPUT_SIZE_1 - KERNEL_SIZE) / STRIDE + 1)
#define OUTPUT_SIZE_3 ((OUTPUT_SIZE_2 - KERNEL_SIZE) / STRIDE + 1)
#define OUTPUT_SIZE_4 ((OUTPUT_SIZE_3 - KERNEL_SIZE) / STRIDE + 1)
#define INPUT_CHANNELS 3
#define NUM_FILTERS_1 16
#define NUM_FILTERS_2 32
#define NUM_FILTERS_3 64
#define NUM_FILTERS_4 128
#define FC_INPUT_SIZE 2048  // (4 * 4 * 128)
#define FC_OUTPUT_SIZE 32   // z_mean & z_log_var
#define DEC_FC_OUTPUT 2048
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
void vae_model(
    volatile fixed_t *input,
    fixed_t kernel_1[NUM_FILTERS_1][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    fixed_t kernel_2[NUM_FILTERS_2][NUM_FILTERS_1][KERNEL_SIZE][KERNEL_SIZE],
    fixed_t kernel_3[NUM_FILTERS_3][NUM_FILTERS_2][KERNEL_SIZE][KERNEL_SIZE],
    fixed_t kernel_4[NUM_FILTERS_4][NUM_FILTERS_3][KERNEL_SIZE][KERNEL_SIZE],
    fixed_t bias_1[NUM_FILTERS_1],
    fixed_t bias_2[NUM_FILTERS_2],
    fixed_t bias_3[NUM_FILTERS_3],
    fixed_t bias_4[NUM_FILTERS_4],
    fixed_t fc_weight_1[FC_OUTPUT_SIZE][FC_INPUT_SIZE],
    fixed_t fc_weight_2[FC_OUTPUT_SIZE][FC_INPUT_SIZE],
    fixed_t fc_bias_1[FC_OUTPUT_SIZE],
    fixed_t fc_bias_2[FC_OUTPUT_SIZE],
	fixed_t dec_dense_weight[DEC_FC_OUTPUT][DEC_FC_INPUT],
	fixed_t dec_dense_bias[DEC_FC_OUTPUT],
	fixed_t dec_kernel_1[DEC_NUM_FILTERS_1][NUM_FILTERS_4][KERNEL_SIZE][KERNEL_SIZE],
	fixed_t dec_bias_1[DEC_NUM_FILTERS_1],
	fixed_t dec_kernel_2[DEC_NUM_FILTERS_2][NUM_FILTERS_3][KERNEL_SIZE][KERNEL_SIZE],
	fixed_t dec_bias_2[DEC_NUM_FILTERS_2],
	fixed_t dec_kernel_3[DEC_NUM_FILTERS_3][DEC_NUM_FILTERS_2][KERNEL_SIZE][KERNEL_SIZE],
	fixed_t dec_bias_3[DEC_NUM_FILTERS_3],
	fixed_t dec_kernel_4[INPUT_CHANNELS][DEC_NUM_FILTERS_3][KERNEL_SIZE][KERNEL_SIZE],
	fixed_t dec_bias_4[INPUT_CHANNELS],
    volatile fixed_t *epsilon_input,
    volatile fixed_t *z_output);

void save_fc_output_to_file(const char *filename, fixed_t *output, int size) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Lá»—i má»Ÿ file: %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        fprintf(file, "%.14f\n", (float)output[i]);
    }

    fclose(file);
}

void print_fc_computation(fixed_t input[FC_INPUT_SIZE], fixed_t weights[FC_OUTPUT_SIZE][FC_INPUT_SIZE], fixed_t bias[FC_OUTPUT_SIZE], fixed_t output[FC_OUTPUT_SIZE]) {
    printf("=== Fully Connected Layer Computation ===\n");

    for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
        printf("Neuron %d:\n", i + 1);

        fixed_t sum = 0;
        for (int j = 0; j < FC_INPUT_SIZE; j++) {
            fixed_t partial = input[j] * weights[i][j];
            sum += partial;
            printf("  input[%d] * weight[%d][%d] = %.8f * %.8f = %.8f\n",
                   j, i, j, (float)input[j], (float)weights[i][j], (float)partial);
        }

        sum += bias[i];
        printf("  + bias[%d] = %.8f\n", i, (float)bias[i]);
        printf("  => Output[%d] = %.8f\n\n", i, (float)sum);

        output[i] = sum;
    }
}

void load_data_from_file_fixed(const char *filename, fixed_t *data, size_t size_bytes) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    int num_elements = size_bytes / sizeof(fixed_t);
    for (int i = 0; i < num_elements; i++) {
        double tmp;
        if (fscanf(file, "%lf", &tmp) != 1) {
            printf("Error reading element %d from file %s\n", i, filename);
            exit(1);
        }

        data[i] = tmp;
    }
    fclose(file);
}

void print_input_channels_fixed(fixed_t *input, int channels, int rows, int cols) {
    for (int c = 0; c < channels; c++) {
        printf("Channel %d:\n", c + 1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%.8f ", (float)input[c * rows * cols + i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_full_output_fixed(fixed_t *output, int filters, int size) {
    for (int f = 0; f < filters; f++) {
        printf("Filter %d Output:\n", f + 1);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int index = f * size * size + i * size + j;
                printf("%.14f ", (float)output[index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_weights_fixed(fixed_t* kernel, int num_filters, int num_channels, int kernel_size) {
    for (int f = 0; f < num_filters; f++) {
        printf("Filter %d Weights:\n", f + 1);
        for (int c = 0; c < num_channels; c++) {
            printf("  Channel %d:\n", c + 1);
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int index = f * num_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + i * kernel_size + j;
                    printf("%.14f ", (float)kernel[index]);
                }
                printf("\n");
            }
        }
        printf("\n");
    }
}

void print_biases_fixed(fixed_t *bias, int num_filters) {
    for (int f = 0; f < num_filters; f++) {
        printf("Filter %d Bias: %.14f\n", f + 1, (float)bias[f]);
    }
}
//IN cac weight bias cua cac Dense(FC) Layer
void print_fc_weights_fixed(fixed_t weights[FC_OUTPUT_SIZE][FC_INPUT_SIZE]) {
    printf("FC Layer Weights:\n");
    for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
        printf("Neuron %d: ", i + 1);
        for (int j = 0; j < FC_INPUT_SIZE; j++) {
            printf("%.8f ", (float)weights[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
void print_fc_weights_fixed1(fixed_t weights[DEC_FC_OUTPUT][DEC_FC_INPUT]) {
    printf("Dec FC Layer Weights:\n");
    for (int i = 0; i < DEC_FC_OUTPUT; i++) {
        printf("Dec Neuron %d: ", i + 1);
        for (int j = 0; j < DEC_FC_INPUT; j++) {
            printf("%.14f ", (float)weights[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
void print_fc_bias_fixed(fixed_t bias[FC_OUTPUT_SIZE]) {
    printf("FC Layer Bias:\n");
    for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
        printf("Bias %d: %.8f\n", i + 1, (float)bias[i]);
    }
    printf("\n");
}
void print_fc_bias_fixed1(fixed_t bias[DEC_FC_OUTPUT]) {
    printf("Dec FC Layer Bias:\n");
    for (int i = 0; i < DEC_FC_OUTPUT; i++) {
        printf("Dec Bias %d: %.14f\n", i + 1, (float)bias[i]);
    }
    printf("\n");
}
void print_fc_output_fixed(const char *label, fixed_t *output, int size) {
    printf("%s:\n", label);
    for (int i = 0; i < size; i++) {
        printf("%.14f ", (float)output[i]);
    }
    printf("\n\n");
}


//---------------------------------------------------------------------
// Main Testbench
//---------------------------------------------------------------------
int main() {
    fixed_t input[INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE];
    fixed_t kernel_1[NUM_FILTERS_1][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    fixed_t kernel_2[NUM_FILTERS_2][NUM_FILTERS_1][KERNEL_SIZE][KERNEL_SIZE];
    fixed_t kernel_3[NUM_FILTERS_3][NUM_FILTERS_2][KERNEL_SIZE][KERNEL_SIZE];
    fixed_t kernel_4[NUM_FILTERS_4][NUM_FILTERS_3][KERNEL_SIZE][KERNEL_SIZE];
    fixed_t bias_1[NUM_FILTERS_1];
    fixed_t bias_2[NUM_FILTERS_2];
    fixed_t bias_3[NUM_FILTERS_3];
    fixed_t bias_4[NUM_FILTERS_4];
    fixed_t output[NUM_FILTERS_4 * OUTPUT_SIZE_4 * OUTPUT_SIZE_4];
    fixed_t fc_weight_1[FC_OUTPUT_SIZE][FC_INPUT_SIZE];
    fixed_t fc_weight_2[FC_OUTPUT_SIZE][FC_INPUT_SIZE];
    fixed_t fc_bias_1[FC_OUTPUT_SIZE];
    fixed_t fc_bias_2[FC_OUTPUT_SIZE];
    fixed_t epsilon[FC_OUTPUT_SIZE];
    fixed_t z_sampling_out[FC_OUTPUT_SIZE];
    fixed_t dense_weight[DEC_FC_OUTPUT][DEC_FC_INPUT];
    fixed_t dense_bias[DEC_FC_OUTPUT];
    fixed_t dense_out[DEC_FC_OUTPUT];
    fixed_t dec_kernel_1[DEC_NUM_FILTERS_1][NUM_FILTERS_4][KERNEL_SIZE][KERNEL_SIZE];
    fixed_t dec_bias_1[DEC_NUM_FILTERS_1];
    fixed_t dec_out_1[DEC_NUM_FILTERS_1 * OUTPUT_SIZE_4 * OUTPUT_SIZE_4];
    fixed_t dec_kernel_2[DEC_NUM_FILTERS_2][NUM_FILTERS_3][KERNEL_SIZE][KERNEL_SIZE];
    fixed_t dec_bias_2[DEC_NUM_FILTERS_2];
    fixed_t dec_out_2[DEC_NUM_FILTERS_2 * DEC_TRANSPOSE_2 * DEC_TRANSPOSE_2];
    fixed_t dec_kernel_3[DEC_NUM_FILTERS_3][NUM_FILTERS_2][KERNEL_SIZE][KERNEL_SIZE];
    fixed_t dec_bias_3[DEC_NUM_FILTERS_3];
    fixed_t dec_out_3[DEC_NUM_FILTERS_3 * DEC_TRANSPOSE_3 * DEC_TRANSPOSE_3];
	fixed_t dec_kernel_4[INPUT_CHANNELS][DEC_NUM_FILTERS_3][KERNEL_SIZE][KERNEL_SIZE];
	fixed_t dec_bias_4[INPUT_CHANNELS];
    fixed_t dec_out_4[INPUT_CHANNELS * DEC_TRANSPOSE_4 * DEC_TRANSPOSE_4];
    const char *fc_weight1_file = "../../../../data/encoder/z_mean_weights.txt";
    const char *fc_weight2_file = "../../../../data/encoder/z_log_var_weights.txt";
    const char *fc_bias1_file = "../../../../data/encoder/z_mean_bias.txt";
    const char *fc_bias2_file = "../../../../data/encoder/z_log_var_bias.txt";
    const char *input_file = "../../../../data/input16_14.txt";
    const char *kernel1_file = "../../../../data/encoder/conv2d_weights.txt";
    const char *kernel2_file = "../../../../data/encoder/conv2d_1_weights.txt";
    const char *kernel3_file = "../../../../data/encoder/conv2d_2_weights.txt";
    const char *kernel4_file = "../../../../data/encoder/conv2d_3_weights.txt";
    const char *bias1_file   = "../../../../data/encoder/conv2d_bias.txt";
    const char *bias2_file   = "../../../../data/encoder/conv2d_1_bias.txt";
    const char *bias3_file   = "../../../../data/encoder/conv2d_2_bias.txt";
    const char *bias4_file   = "../../../../data/encoder/conv2d_3_bias.txt";
    const char *epsilon_file = "../../../../data/decoder/epsilon2.txt";
    const char *dec_dense_weight_file = "../../../../data/decoder/dense_weight.txt";
    const char *dec_dense_bias_file = "../../../../data/decoder/dense_bias.txt";
    const char *dec_kernel_1_file = "../../../../data/decoder/conv2d_transpose_weight.txt";
    const char *dec_bias_1_file = "../../../../data/decoder/conv2d_transpose_bias.txt";
    const char *dec_kernel_2_file = "../../../../data/decoder/conv2d_transpose_1_weight.txt";
    const char *dec_bias_2_file = "../../../../data/decoder/conv2d_transpose_1_bias.txt";
    const char *dec_kernel_3_file = "../../../../data/decoder/conv2d_transpose_2_weight.txt";
    const char *dec_bias_3_file = "../../../../data/decoder/conv2d_transpose_2_bias.txt";
    const char *dec_kernel_4_file = "../../../../data/decoder/conv2d_transpose_3_weight.txt";
    const char *dec_bias_4_file = "../../../../data/decoder/conv2d_transpose_3_bias.txt";
    load_data_from_file_fixed(input_file, input, sizeof(input));
    load_data_from_file_fixed(kernel1_file, (fixed_t*)kernel_1, sizeof(kernel_1));
    load_data_from_file_fixed(kernel2_file, (fixed_t*)kernel_2, sizeof(kernel_2));
    load_data_from_file_fixed(kernel3_file, (fixed_t*)kernel_3, sizeof(kernel_3));
    load_data_from_file_fixed(kernel4_file, (fixed_t*)kernel_4, sizeof(kernel_4));
    load_data_from_file_fixed(bias1_file, bias_1, sizeof(bias_1));
    load_data_from_file_fixed(bias2_file, bias_2, sizeof(bias_2));
    load_data_from_file_fixed(bias3_file, bias_3, sizeof(bias_3));
    load_data_from_file_fixed(bias4_file, bias_4, sizeof(bias_4));
    load_data_from_file_fixed(fc_weight1_file, (fixed_t*)fc_weight_1, sizeof(fc_weight_1));
    load_data_from_file_fixed(fc_weight2_file, (fixed_t*)fc_weight_2, sizeof(fc_weight_2));
    load_data_from_file_fixed(fc_bias1_file, fc_bias_1, sizeof(fc_bias_1));
    load_data_from_file_fixed(fc_bias2_file, fc_bias_2, sizeof(fc_bias_2));
    load_data_from_file_fixed(epsilon_file, epsilon, sizeof(epsilon));
    load_data_from_file_fixed(dec_dense_weight_file, (fixed_t*)dense_weight, sizeof(dense_weight));
    load_data_from_file_fixed(dec_dense_bias_file, dense_bias, sizeof(dense_bias));
    load_data_from_file_fixed(dec_kernel_1_file, (fixed_t*)dec_kernel_1, sizeof(dec_kernel_1));
    load_data_from_file_fixed(dec_bias_1_file, dec_bias_1, sizeof(dec_bias_1));
    load_data_from_file_fixed(dec_kernel_2_file, (fixed_t*)dec_kernel_2, sizeof(dec_kernel_2));
    load_data_from_file_fixed(dec_bias_2_file, dec_bias_2, sizeof(dec_bias_2));
    load_data_from_file_fixed(dec_kernel_3_file, (fixed_t*)dec_kernel_3, sizeof(dec_kernel_3));
    load_data_from_file_fixed(dec_bias_3_file, dec_bias_3, sizeof(dec_bias_3));
    load_data_from_file_fixed(dec_kernel_4_file, (fixed_t*)dec_kernel_4, sizeof(dec_kernel_4));
    load_data_from_file_fixed(dec_bias_4_file, dec_bias_4, sizeof(dec_bias_4));
    printf("Input Channels:\n");
    print_input_channels_fixed(input, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE);

    vae_model(
        (volatile fixed_t*)input,
        kernel_1,
        kernel_2,
        kernel_3,
        kernel_4,
        bias_1,
        bias_2,
        bias_3,
        bias_4,
		fc_weight_1,
		fc_weight_2,
		fc_bias_1,
		fc_bias_2,
		dense_weight,
		dense_bias,
		dec_kernel_1,
		dec_bias_1,
		dec_kernel_2,
		dec_bias_2,
		dec_kernel_3,
		dec_bias_3,
		dec_kernel_4,
		dec_bias_4,
        (volatile fixed_t*)epsilon,
        (volatile fixed_t*)dec_out_4
    );

    printf("Convolution Transpose 4 Weights:\n");
    print_weights_fixed((fixed_t*)dec_kernel_4, DEC_NUM_FILTERS_4, DEC_NUM_FILTERS_3, KERNEL_SIZE);
    print_biases_fixed((fixed_t*)dec_bias_4, DEC_NUM_FILTERS_4);
    print_full_output_fixed(dec_out_4, DEC_NUM_FILTERS_4, DEC_TRANSPOSE_4);
    save_fc_output_to_file("C:\\Users\\ACER\\Downloads\\out_transpose_4.txt", dec_out_4, DEC_NUM_FILTERS_4 * DEC_TRANSPOSE_4 * DEC_TRANSPOSE_4);
    return 0;
}
