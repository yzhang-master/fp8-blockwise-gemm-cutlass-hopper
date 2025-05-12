#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "helper.h"
#include <cublas_v2.h>

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = float;
using ElementD = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
constexpr int AlignmentAB = 16;
constexpr int AlignmentCD = 4;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _2, _1>;
using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(TileShape{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, ElementC, LayoutC, AlignmentCD, ElementD, LayoutD, AlignmentCD,
    EpilogueSchedule>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentAB,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentAB,
    float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int warmup = 10;
  int iterations = 50;
  bool verify = true;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iterations", iterations);
    if (cmd.check_cmd_line_flag("no-verify")) verify = false;
    if (m % 128 || n % 128 || k % 128 || m <= 0 || n <= 0 || k <= 0) {
      throw std::invalid_argument("M, N and K must be positive multiples of 128");
    }
    if (warmup < 0 || iterations <= 0) {
      throw std::invalid_argument("iterations must be positive and warmup non-negative");
    }
  }
};

struct Tensors {
  using VectorLayout = cutlass::layout::PackedVectorLayout;
  cutlass::HostTensor<ElementA, LayoutA> a;
  cutlass::HostTensor<ElementB, LayoutB> b;
  cutlass::HostTensor<float, VectorLayout> sfa;
  cutlass::HostTensor<float, VectorLayout> sfb;
  cutlass::HostTensor<ElementC, LayoutC> c;
  cutlass::HostTensor<ElementD, LayoutD> d;
  cutlass::HostTensor<ElementD, LayoutD> reference;
  Gemm::Arguments arguments;
  Gemm::GemmKernel::StrideA stride_a;
  Gemm::GemmKernel::StrideB stride_b;
  Gemm::GemmKernel::StrideC stride_c;
  Gemm::GemmKernel::StrideD stride_d;
  LayoutSFA layout_sfa;
  LayoutSFB layout_sfb;

  explicit Tensors(Options const& options) {
    stride_a = cutlass::make_cute_packed_stride<Gemm::GemmKernel::StrideA>({options.m, options.k, 1});
    stride_b = cutlass::make_cute_packed_stride<Gemm::GemmKernel::StrideB>({options.n, options.k, 1});
    stride_c = cutlass::make_cute_packed_stride<Gemm::GemmKernel::StrideC>({options.m, options.n, 1});
    stride_d = cutlass::make_cute_packed_stride<Gemm::GemmKernel::StrideD>({options.m, options.n, 1});
    layout_sfa = ScaleConfig::tile_atom_to_shape_SFA({options.m, options.n, options.k, 1});
    layout_sfb = ScaleConfig::tile_atom_to_shape_SFB({options.m, options.n, options.k, 1});
    a.resize({options.m, options.k});
    b.resize({options.k, options.n});
    c.resize({options.m, options.n});
    d.resize({options.m, options.n});
    reference.resize({options.m, options.n});
    sfa.resize({size(filter_zeros(layout_sfa))});
    sfb.resize({size(filter_zeros(layout_sfb))});
    cutlass::reference::host::TensorFillRandomUniform(a.host_view(), 17, 2.0, -2.0, 8);
    cutlass::reference::host::TensorFillRandomUniform(b.host_view(), 23, 2.0, -2.0, 8);
    cutlass::reference::host::TensorFillRandomUniform(sfa.host_view(), 29, 0.02, 0.005, 0);
    cutlass::reference::host::TensorFillRandomUniform(sfb.host_view(), 31, 0.02, 0.005, 0);
    cutlass::reference::host::TensorFill(c.host_view(), 0.0f);
    a.sync_device();
    b.sync_device();
    sfa.sync_device();
    sfb.sync_device();
    c.sync_device();
    d.sync_device();
    arguments = Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k, 1},
        {a.device_data(), stride_a, b.device_data(), stride_b,
         sfa.device_data(), layout_sfa, sfb.device_data(), layout_sfb},
        {{1.0f, 0.0f}, c.device_data(), stride_c, d.device_data(), stride_d}};
  }

  bool verify(Options const& options) {
    auto tensor_a = make_tensor(a.host_data(), make_layout(make_shape(options.m, options.k, 1), stride_a));
    auto tensor_b = make_tensor(b.host_data(), make_layout(make_shape(options.n, options.k, 1), stride_b));
    auto tensor_sfa = make_tensor(sfa.host_data(), layout_sfa);
    auto tensor_sfb = make_tensor(sfb.host_data(), layout_sfb);
    auto tensor_c = make_tensor(c.host_data(), make_layout(make_shape(options.m, options.n, 1), stride_c));
    auto tensor_d = make_tensor(reference.host_data(), make_layout(make_shape(options.m, options.n, 1), stride_d));
    cutlass::reference::host::GettBlockScalingMainloopParams<
        float, decltype(tensor_a), decltype(tensor_sfa), decltype(tensor_b), decltype(tensor_sfb)>
        mainloop{tensor_a, tensor_sfa, tensor_b, tensor_sfb};
    cutlass::reference::host::GettEpilogueParams<
        float, float, float, float, decltype(tensor_c), decltype(tensor_d)>
        epilogue{1.0f, 0.0f, tensor_c, tensor_d};
    cutlass::reference::host::Gemm3x(mainloop, epilogue);
    d.sync_host();
    return cutlass::reference::host::TensorRelativelyEquals(
        d.host_view(), reference.host_view(), 1e-4f, 1e-3f);
  }
};

void check_cublas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS call failed");
}

double benchmark_cublas_bf16(Options const& options) {
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle));
  cutlass::device_memory::allocation<cutlass::bfloat16_t> a(size_t(options.m) * options.k);
  cutlass::device_memory::allocation<cutlass::bfloat16_t> b(size_t(options.k) * options.n);
  cutlass::device_memory::allocation<float> d(size_t(options.m) * options.n);
  float alpha = 1.0f;
  float beta = 0.0f;
  auto launch = [&]() {
    check_cublas(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, options.m, options.n, options.k,
        &alpha, a.get(), CUDA_R_16BF, options.k, b.get(), CUDA_R_16BF, options.n,
        &beta, d.get(), CUDA_R_32F, options.m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  for (int i = 0; i < options.warmup; ++i) launch();
  GpuTimer timer;
  timer.start();
  for (int i = 0; i < options.iterations; ++i) launch();
  timer.stop();
  check_cublas(cublasDestroy(handle));
  return timer.elapsed_millis() / options.iterations;
}

int run(Options const& options) {
  Tensors tensors(options);
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(tensors.arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(tensors.arguments));
  CUTLASS_CHECK(gemm.initialize(tensors.arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (options.verify && !tensors.verify(options)) {
    std::cerr << "CUTLASS output differs from the block-scaled host reference\n";
    return 3;
  }
  for (int i = 0; i < options.warmup; ++i) CUTLASS_CHECK(gemm.run());
  GpuTimer timer;
  timer.start();
  for (int i = 0; i < options.iterations; ++i) CUTLASS_CHECK(gemm.run());
  timer.stop();
  double fp8_ms = timer.elapsed_millis() / options.iterations;
  double bf16_ms = benchmark_cublas_bf16(options);
  double flops = 2.0 * options.m * options.n * options.k;
  std::cout << "verified=" << (options.verify ? "true" : "false") << "\n"
            << "m=" << options.m << " n=" << options.n << " k=" << options.k << "\n"
            << "fp8_blockwise_ms=" << fp8_ms << "\n"
            << "fp8_blockwise_tflops=" << flops / fp8_ms * 1e-9 << "\n"
            << "cublas_bf16_ms=" << bf16_ms << "\n"
            << "cublas_bf16_tflops=" << flops / bf16_ms * 1e-9 << "\n";
  return 0;
}
#endif

int main(int argc, char const** argv) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 9) {
    std::cerr << "This executable requires Hopper SM90; detected compute capability "
              << properties.major << "." << properties.minor << "\n";
    return 2;
  }
  try {
    Options options;
    options.parse(argc, argv);
    return run(options);
  } catch (std::exception const& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
#else
  std::cerr << "Rebuild with SM90 MMA support enabled\n";
  return 2;
#endif
}
