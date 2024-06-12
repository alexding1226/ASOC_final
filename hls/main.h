#ifndef _MAIN__H
#define _MAIN__H

#include <ac_int.h>
#include <ac_channel.h>
#include <mc_scverify.h>
#include <ac_fixed.h>
#include <conv2d.h>
#include <maxpool.h>
#include <batchnorm_relu.h>
#include <tranconv2d.h>
#include <defs.h>



#pragma hls_design top
class main{

    //instances
    conv2d conv2d_inst;
    batchnorm_relu batchnorm_relu_inst;
    maxpool maxpool_inst;
    tranconv2d tranconv2d_inst;

    // filter memory
    // filterType filters[485120];
    // filterType gamma[736];
    // filterType beta[736];

    // buffer memory
    bufType buf1[64*64*8];
    bufType padded_input[66*66*3];
    bufType buf2[64*64*8];

    // concat buffer memory
    bufType enc1 [64*64*8];
    bufType enc2 [32*32*16];
    bufType enc3 [16*16*32];
    bufType enc4 [8*8*64],;


    public:
    main() {}
    #pragma hls_design interface
     void CCS_BLOCK(run)(
                        ac_channel<bufType> &input,
                        ac_channel<bufType> &output,
                        ) 
    {        
    
    ac_int <8, false> i;
    for (i = 0; i<3*64*64; i++){
        buf1[i] = input.read();
    }

    ac_int<20, false> filter_offset = 0;
    ac_int<10, false> gamma_offset = 0;
    //enc1

    //conv2d : input, padded_input, output, filters, height, width, kernel_size, padding, filter_offset, in_channels, out_channels
    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 64, 64, 3, 1, 0, 3, 8);
    //batchnorm_relu : input, output, gamma, beta, channels, height, width, offset
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 8, 64, 64, 0);
    filter_offset += 8*3*3*3;
    gamma_offset += 8;

    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 64, 64, 3, 1, filter_offset, 8, 8);
    batchnorm_relu_inst.run(buf2, enc1, gamma_pretrain, beta_pretrain, 8, 64, 64, gamma_offset);
    filter_offset += 8*8*3*3;
    gamma_offset += 8;
    //maxpool : input, output,channels, height, width,  pool_size, stride
    maxpool_inst.run(enc1, buf1, 8, 64, 64, 2, 2);

    //enc2
    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 32, 32, 3, 1, filter_offset, 8, 16);
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 16, 32, 32, gamma_offset);
    filter_offset += 16*8*3*3;
    gamma_offset += 16;

    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 32, 32, 3, 1, filter_offset, 16, 16);
    batchnorm_relu_inst.run(buf2, enc2, gamma_pretrain, beta_pretrain, 16, 32, 32, gamma_offset);
    filter_offset += 16*16*3*3;
    gamma_offset += 16;
    maxpool_inst.run(enc2, buf1, 16, 32, 32, 2, 2);

    //enc3
    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 16, 16, 3, 1, filter_offset, 16, 32);
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 32, 16, 16, gamma_offset);
    filter_offset += 32*16*3*3;
    gamma_offset += 32;

    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 16, 16, 3, 1, filter_offset, 32, 32);
    batchnorm_relu_inst.run(buf2, enc3, gamma_pretrain, beta_pretrain, 32, 16, 16, gamma_offset);
    filter_offset += 32*32*3*3;
    gamma_offset += 32;
    maxpool_inst.run(enc3, buf1, 32, 16, 16, 2, 2);

    //enc4
    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 8, 8, 3, 1, filter_offset, 32, 64);
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 64, 8, 8, gamma_offset);
    filter_offset += 64*32*3*3;
    gamma_offset += 64;

    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 8, 8, 3, 1, filter_offset, 64, 64);
    batchnorm_relu_inst.run(buf2, enc4, gamma_pretrain, beta_pretrain, 64, 8, 8, gamma_offset);
    filter_offset += 64*64*3*3;
    gamma_offset += 64;
    maxpool_inst.run(enc4, buf1, 64, 8, 8, 2, 2);

    //bottleneck
    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 4, 4, 3, 1, filter_offset, 64, 128);
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 128, 4, 4, gamma_offset);
    filter_offset += 128*64*3*3;
    gamma_offset += 128;

    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 4, 4, 3, 1, filter_offset, 128, 128);
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 128, 4, 4, gamma_offset);
    filter_offset += 128*128*3*3;
    gamma_offset += 128;

    //dec4
    //tranconv2d : input, output, filters, height, width, kernel_size, filter_offset, in_channels, out_channels, stride
    tranconv2d_inst.run(buf1, buf2, filters_pretrain, 4, 4, 3, filter_offset, 128, 64, 2);
    filter_offset += 64*128*2*2;
    //concat
    for (i = 8*8*64; i<8*8*64*2; i++){
        buf2[i] = enc4[i-8*8*64];
    }
    //conv2d : input, padded_input, output, filters, height, width, kernel_size, padding, filter_offset, in_channels, out_channels
    conv2d_inst.run(buf2, padded_input, buf1, filters_pretrain, 8, 8, 3, 1, filter_offset, 128, 64);
    batchnorm_relu_inst.run(buf1, buf2, gamma_pretrain, beta_pretrain, 64, 8, 8, gamma_offset);
    filter_offset += 64*128*3*3;
    gamma_offset += 64;

    conv2d_inst.run(buf2, padded_input, buf1, filters_pretrain, 8, 8, 3, 1, filter_offset, 64, 64);
    batchnorm_relu_inst.run(buf1, buf2, gamma_pretrain, beta_pretrain, 64, 8, 8, gamma_offset);
    filter_offset += 64*64*3*3;
    gamma_offset += 64;

    //dec3
    tranconv2d_inst.run(buf2, buf1, filters_pretrain, 8, 8, 3, filter_offset, 64, 32, 2);
    for (i = 16*16*32; i<16*16*32*2; i++){
        buf1[i] = enc3[i-16*16*32];
    }
    filter_offset += 32*64*2*2;
    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 16, 16, 3, 1, filter_offset, 64, 32);
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 32, 16, 16, gamma_offset);
    filter_offset += 32*64*3*3;
    gamma_offset += 32;

    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 16, 16, 3, 1, filter_offset, 32, 32);
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 32, 16, 16, gamma_offset);
    filter_offset += 32*32*3*3;
    gamma_offset += 32;

    //dec2
    tranconv2d_inst.run(buf1, buf2, filters_pretrain, 16, 16, 3, filter_offset, 32, 16, 2);
    for (i = 32*32*16; i<32*32*16*2; i++){
        buf2[i] = enc2[i-32*32*16];
    }
    filter_offset += 16*32*2*2;
    conv2d_inst.run(buf2, padded_input, buf1, filters_pretrain, 32, 32, 3, 1, filter_offset, 32, 16);
    batchnorm_relu_inst.run(buf1, buf2, gamma_pretrain, beta_pretrain, 16, 32, 32, gamma_offset);
    filter_offset += 16*32*3*3;
    gamma_offset += 16;

    conv2d_inst.run(buf2, padded_input, buf1, filters_pretrain, 32, 32, 3, 1, filter_offset, 16, 16);
    batchnorm_relu_inst.run(buf1, buf2, gamma_pretrain, beta_pretrain, 16, 32, 32, gamma_offset);
    filter_offset += 16*16*3*3;
    gamma_offset += 16;

    //dec1
    tranconv2d_inst.run(buf2, buf1, filters_pretrain, 32, 32, 3, filter_offset, 16, 8, 2);
    for (i = 64*64*8; i<64*64*8*2; i++){
        buf1[i] = enc1[i-64*64*8];
    }
    filter_offset += 8*16*2*2;
    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 64, 64, 3, 1, filter_offset, 16, 8);
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 8, 64, 64, gamma_offset);
    filter_offset += 8*16*3*3;
    gamma_offset += 8;

    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 64, 64, 3, 1, filter_offset, 8, 8);
    batchnorm_relu_inst.run(buf2, buf1, gamma_pretrain, beta_pretrain, 8, 64, 64, gamma_offset);
    filter_offset += 8*8*3*3;
    gamma_offset += 8;

    //output
    conv2d_inst.run(buf1, padded_input, buf2, filters_pretrain, 64, 64, 3, 1, filter_offset, 8, 21);

    for (i = 0; i<64*64*21; i++){
        output.write(buf2[i]);
    }
    }
  

};

#endif