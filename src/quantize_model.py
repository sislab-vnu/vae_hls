import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

from vae_model import VAE
from fxpmath import Fxp
fxp_ref_1 = Fxp(None, dtype='fxp-s16/14')

def load_model_from_files(encoder_file, decoder_file):
    encoder = load_model(encoder_file)
    decoder = load_model(decoder_file)

    vae_f = VAE(encoder, decoder, beta=0.001)

    return vae_f
def update_weights(model):
    w_dict = {}
    extract_weights_recursive(model, w_dict)
    w_fxp_dict = {}
    # Lấy một lớp đầu tiên có trọng số và bias trong w_dict
    for layer_name, params in w_dict.items():
        if params["weights"] is not None:  # Kiểm tra nếu lớp này có weights
            print(f"Layer: {layer_name}")
            print(f"  Weights: \n{params['weights']}")  # In toàn bộ giá trị của weights
            print(f"  Weights Mean: {np.mean(params['weights']):.4f}, Weights Std: {np.std(params['weights']):.4f}")  # Mean và Std của weights
            print(f"  Bias: \n{params['bias']}")       # In toàn bộ giá trị của bias
            print(f"  Bias Mean: {np.mean(params['bias']):.4f}, Bias Std: {np.std(params['bias']):.4f}")  # Mean và Std của bias
            break  # Dừng sau khi in thông tin lớp đầu tiên
    for layer in w_dict.keys():
        w_fxp_dict[layer] = [
            Fxp(w_dict[layer]["weights"], like=fxp_ref_1),  # Chuyển đổi weights sang fixed-point
            Fxp(w_dict[layer]["bias"], like=fxp_ref_1),     # Chuyển đổi bias sang fixed-point
        ]

    update_weights_in_model(model, w_fxp_dict)
def write_file(fname, data):
    with open(fname, 'w') as f:
        f.write('# Array shape: {0}\n'.format(data.shape))
        if len(data.shape) > 2:
            kernel_h, kernel_w, input_channels, output_channels = data.shape
            for o in range(0,output_channels):
                for i in range(0,input_channels):
                    np.savetxt(f, data[:,:,i,o], fmt='%-7.8f')
                    f.write('# New slice: input_channel=%d output_channel=%d\n' %(i,o))
        else:
            np.savetxt(f, data, fmt='%-7.8f')
    f.close()
    
def dump_weights(model):
    for l in model.layers:
        if hasattr(l, 'layers'):
            print("Sub layers exist")
        else:
            d = l.get_weights()
            if len(d) > 0:
                print(l.name, len(d))
                weights = d[0]
                bias = d[1]
                write_file("_".join([l.name, "_weights.txt"]), weights)
                write_file("_".join([l.name, "_bias.txt"]), bias)
                
                
    
def quantize_model(encoder_file, decoder_file):
    vae = load_model_from_files(encoder_file, decoder_file)
    vae.compile()
    dump_weights(vae.encoder)

if __name__ == "__main__":
    import sys
    quantize_model(sys.argv[1], sys.argv[2])
