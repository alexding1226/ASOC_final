#ifndef _BATCHRELU__H
#define _BATCHRELU__H

#include <ac_int.h>
#include <ac_channel.h>
#include <mc_scverify.h>
#include <ac_fixed.h>

class batchnorm_relu{

    public:
    batchnorm_relu() {}
    #pragma hls_design interface
     void CCS_BLOCK(run)(
                        ac_fixed<8, 4, true> input[64*64*8], // memory interface
                        ac_fixed<8, 4, true> output[64*64*8],
                        ac_fixed<4, 8, true> gamma[736],
                        ac_fixed<4, 8, true> beta[736],
                        ac_int<7, false> &channels, // direct input
                        ac_int<7, false> &height, // direct input
                        ac_int<7, false> &width // direct input
                        ) 
    {        
    ac_int<13, false> num_elements = height * width;
    ac_fixed<0,17,false> epsilon = 1e-5;

    for (ac_int<8, false> c = 0; c < channels; c++) {
        ac_fixed<14,8,false>  mean     = 0;
        ac_fixed<22,16,false> var      = 0;
        ac_fixed<20,8,false> mean_tmp = 0;
        ac_fixed<34,16,false> var_tmp  = 0;
        ac_int<13, false> i;
        

        // Calculate mean
        for (i = 0; i < num_elements; i++) {
            mean_tmp += input[c * num_elements + i];
        }
        mean = mean_tmp / num_elements;

        // Calculate variance
        for (i = 0; i < num_elements; i++) {
            var_tmp += (input[c * num_elements + i] - mean) * (input[c * num_elements + i] - mean);
        }
        var= var_tmp / num_elements;

        ac_int<20, false> idx;
        ac_fixed<14,8,false> var_sqrt;
        ac_fixed<22,17,false> sum;
        ac_fixed<22,16,false> out_tmp;
        // Normalize, scale, and shift
        for (i = 0; i < num_elements; i++) {
            idx = c * num_elements + i;
            sum = var + epsilon;
            out_tmp = gamma[c] * ((input[idx] - mean) / ac_math::ac_sqrt_pwl(sum, var_sqrt)) + beta[c];
            output[idx] = (out_tmp > 0) ? out_tmp : 0;
        }
    }

};

#endif