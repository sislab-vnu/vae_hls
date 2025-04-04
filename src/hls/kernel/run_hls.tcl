#Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

set project_name "vae"
set opt_method "baseline"


# Create a project
open_project -reset ${project_name}

# Add design files
add_files ${project_name}.cpp
# Add test bench & files
add_files -tb ${project_name}_test.cpp
add_files -tb data/encoder/conv2d_1_bias.txt
add_files -tb data/encoder/conv2d_1_weights.txt
add_files -tb data/encoder/conv2d_2_bias.txt
add_files -tb data/encoder/conv2d_2_weights.txt
add_files -tb data/encoder/conv2d_3_bias.txt
add_files -tb data/encoder/conv2d_3_weights.txt
add_files -tb data/encoder/conv2d_bias.txt
add_files -tb data/encoder/conv2d_weights.txt
add_files -tb data/encoder/z_log_var_bias.txt
add_files -tb data/encoder/z_log_var_weights.txt
add_files -tb data/encoder/z_mean_bias.txt
add_files -tb data/encoder/z_mean_weights.txt
add_files -tb data/decoder/conv2d_transpose_1_bias.txt
add_files -tb data/decoder/conv2d_transpose_1_weight.txt
add_files -tb data/decoder/conv2d_transpose_2_bias.txt
add_files -tb data/decoder/conv2d_transpose_2_weight.txt
add_files -tb data/decoder/conv2d_transpose_3_bias.txt
add_files -tb data/decoder/conv2d_transpose_3_weight.txt
add_files -tb data/decoder/conv2d_transpose_bias.txt
add_files -tb data/decoder/conv2d_transpose_weight.txt
add_files -tb data/decoder/dense_bias.txt
add_files -tb data/decoder/dense_weight.txt
add_files -tb data/decoder/epsilon2.txt
add_files -tb data/input16_14.txt
add_files -tb data/out_transpose_4.txt

# Set the top-level function
set_top vae_model

# ########################################################
# Create a solution
open_solution -reset ${opt_method} -flow_target vivado

# Define technology and clock rate
set_part  {xc7z020-clg484-1}
create_clock -period 5

# Set variable to select which steps to execute
set hls_exec 3


csim_design
# Set any optimization directives
# End of directives

if {$hls_exec >= 1} {
	# Run Synthesis
   csynth_design
}
if {$hls_exec >= 2} {
	# Run Synthesis, RTL Simulation
   cosim_design
}
if {$hls_exec >= 3} { 
	# Run Synthesis, RTL Simulation, RTL implementation
   #export_design -format ip_catalog -version "1.00a" -library "hls" -vendor "xilinx.com" -description "A memory mapped IP created by Vitis HLS" -evaluate verilog
   export_design -format ip_catalog -evaluate verilog
}

exit
