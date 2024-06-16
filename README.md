# ASoC Final Project
U-Net for Image Segmentation

## File Hierarchy

### `unet_model/`
- U-Net profiling, training, and evaluation

### `unet_c/`
- Translate program from Python to C

### `hls*/`
- Hardware implementation by high-level synthesis (HLS)

### `fsic_fpga/`
- FSIC integration and simulation

```
$ cd fsic_fpga/rtl/user/testbench/tc
$ ./run_xsim
```

### `doc/`
- Contains report and presentation slides
