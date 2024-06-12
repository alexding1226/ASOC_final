void mc_testbench_capture_IN( ac_fixed<12, 8, true, AC_TRN, AC_WRAP> input[32768], ac_fixed<12, 8, true, AC_TRN, AC_WRAP> output[32768], ac_int<7, false> &height, ac_int<7, false> &width, ac_int<2, false> &kernel_size, ac_int<20, false> &filter_offset, ac_int<7, false> &in_channels, ac_int<7, false> &out_channels, ac_int<3, false> &stride ) {
  mc_testbench::capture_IN(input, output, height, width, kernel_size, filter_offset, in_channels, out_channels, stride);
}
void mc_testbench_capture_OUT( ac_fixed<12, 8, true, AC_TRN, AC_WRAP> input[32768], ac_fixed<12, 8, true, AC_TRN, AC_WRAP> output[32768], ac_int<7, false> &height, ac_int<7, false> &width, ac_int<2, false> &kernel_size, ac_int<20, false> &filter_offset, ac_int<7, false> &in_channels, ac_int<7, false> &out_channels, ac_int<3, false> &stride ) {
  mc_testbench::capture_OUT(input, output, height, width, kernel_size, filter_offset, in_channels, out_channels, stride);
}

class ccs_intercept
{
  public:
  void capture_THIS( void* _this ) {
    cur_inst = _this;
    if ( !dut_inst_set() ) dut = mc_testbench_set_dut_inst(cur_inst);
    if ( capture_msg && dut != NULL && dut == cur_inst ) {
      std::cout << "SCVerify intercepting C++ function 'UNET_IP::tranconv2d::run_tran' for RTL block 'UNET_IP_tranconv2d'" << std::endl;
      std::cout << "                      DUT instance '" << dut << "'" << std::endl;
      capture_msg = false;
    }
  }
  void capture_IN( ac_fixed<12, 8, true, AC_TRN, AC_WRAP> input[32768], ac_fixed<12, 8, true, AC_TRN, AC_WRAP> output[32768], ac_int<7, false> &height, ac_int<7, false> &width, ac_int<2, false> &kernel_size, ac_int<20, false> &filter_offset, ac_int<7, false> &in_channels, ac_int<7, false> &out_channels, ac_int<3, false> &stride ) {
    if ( dut != NULL && dut == cur_inst )
      mc_testbench_capture_IN(input, output, height, width, kernel_size, filter_offset, in_channels, out_channels, stride);
  }
  void capture_OUT( ac_fixed<12, 8, true, AC_TRN, AC_WRAP> input[32768], ac_fixed<12, 8, true, AC_TRN, AC_WRAP> output[32768], ac_int<7, false> &height, ac_int<7, false> &width, ac_int<2, false> &kernel_size, ac_int<20, false> &filter_offset, ac_int<7, false> &in_channels, ac_int<7, false> &out_channels, ac_int<3, false> &stride ) {
    if ( dut != NULL && dut == cur_inst )
      mc_testbench_capture_OUT(input, output, height, width, kernel_size, filter_offset, in_channels, out_channels, stride);
  }
  void wait_for_idle_sync() {
    if ( dut != NULL && dut == cur_inst )
      mc_testbench_wait_for_idle_sync();
  }
  ccs_intercept(): dut(NULL), cur_inst(NULL), capture_msg(true){}
  ~ccs_intercept() {
    if ( capture_msg )
      std::cout << "Error: The CCS_BLOCK intercept did not occur for DUT instance '" << dut << "'" << std::endl
                << "       Make sure the applied pointer is for class 'UNET_IP::tranconv2d'" << std::endl;
  }
  private:
  bool dut_inst_set() {
    if ( dut == NULL && !!mc_testbench_dut_inst() )
      dut = mc_testbench_dut_inst();
    return dut != NULL;
  }
  void *dut, *cur_inst;
  bool capture_msg;
};


namespace UNET_IP {
  void tranconv2d::run_tran(ac_fixed<12, 8, true, AC_TRN, AC_WRAP> input[32768], ac_fixed<12, 8, true, AC_TRN, AC_WRAP> output[32768], ac_int<7, false> &height, ac_int<7, false> &width, ac_int<2, false> &kernel_size, ac_int<20, false> &filter_offset, ac_int<7, false> &in_channels, ac_int<7, false> &out_channels, ac_int<3, false> &stride) {
    static ccs_intercept ccs_intercept_inst;
    ccs_intercept_inst.capture_THIS(this);
    ccs_intercept_inst.wait_for_idle_sync();
    ccs_intercept_inst.capture_IN(input, output, height, width, kernel_size, filter_offset, in_channels, out_channels, stride);
    ccs_real_run_tran(input, output, height, width, kernel_size, filter_offset, in_channels, out_channels, stride);
    ccs_intercept_inst.capture_OUT(input, output, height, width, kernel_size, filter_offset, in_channels, out_channels, stride);
  }
}

