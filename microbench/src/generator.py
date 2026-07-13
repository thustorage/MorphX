import pandas as pd
import os

def generate_so_and_header(csv_file_path, output_dir="."):
    """
    生成用于编译成共享库的 .cu 源文件和对应的 .h 公共头文件。

    Args:
        csv_file_path (str): 输入的 result.csv 文件路径。
        output_dir (str): 输出目录。
    """
    # --- 和之前一样的CSV解析逻辑 ---
    try:
        df = pd.read_csv(csv_file_path, skipinitialspace=True)
    except FileNotFoundError:
        print(f"错误: 文件 '{csv_file_path}' 未找到。")
        return

    successful_runs = df[df['Status'] == 'success'].copy()
    successful_runs['Runtime'] = pd.to_numeric(successful_runs['Runtime'])
    best_configs_idx = successful_runs.groupby(['m', 'n', 'k'])['Runtime'].idxmin()
    best_configs = successful_runs.loc[best_configs_idx].sort_values(['m', 'n', 'k'])

    if best_configs.empty:
        print("警告: CSV文件中无成功记录，无法生成文件。")
        return

    # --- 1. 生成轻量级的 C-style 头文件 (my_cutlass_gemm.h) ---
    
    header_path = os.path.join(output_dir, "my_cutlass_gemm.h")
    header_content = """\
#pragma once

#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

// 为了确保在C++项目中能够正确链接C风格的符号，使用 extern "C"
#ifdef __cplusplus
extern "C" {
#endif

// 前向声明依赖的类型，避免包含完整的CUDA或CUTLASS头文件
typedef struct CUstream_st* cudaStream_t;
namespace cutlass { struct half_t; }

/**
 * @brief 执行半精度矩阵乘法 D = alpha * A * B + beta * C.
 *
 * 该函数内部通过查找表自动选择最优的CUTLASS内核。
 * 实现被编译在独立的共享库中，以加速主项目的编译。
 */
void myCutlassHgemm(
    int m,
    int n,
    int k,
    const cutlass::half_t* a,
    long long int lda, // 使用标准的 long long int 避免依赖 int64_t 定义
    const cutlass::half_t* b,
    long long int ldb,
    cutlass::half_t* c,
    long long int ldc,
    float alpha,
    float beta,
    cudaStream_t stream
);

#ifdef __cplusplus
} // extern "C"
#endif
"""
    with open(header_path, 'w') as f:
        f.write(header_content)
    print(f"成功生成公共头文件: '{header_path}'")


    # --- 2. 生成包含所有实现的 .cu 源文件 (my_cutlass_gemm.cu) ---
    
    cu_path = os.path.join(output_dir, "my_cutlass_gemm.cu")
    
    # 源文件内容从之前脚本的头文件内容大部分迁移过来
    cu_content = """\
// 包含所有必要的重型头文件
#include "my_cutlass_gemm.h" // 包含我们自己的声明以确保函数签名匹配
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <map>
#include <tuple>
#include <cmath>
#include <limits>

// 内部实现命名空间，避免污染全局
namespace my_cutlass_gemm_impl {

// 函数指针类型定义
using GemmFuncPtr = void(*)(
    int m, int n, int k,
    const cutlass::half_t* a, int64_t lda,
    const cutlass::half_t* b, int64_t ldb,
    cutlass::half_t* c, int64_t ldc,
    float alpha, float beta,
    cudaStream_t stream
);

// 模板化的内核执行函数 (和之前一样)
template <
    int T_M, int T_N, int T_K, int W_M, int W_N, int W_K, 
    int I_M, int I_N, int I_K, int Stages
>
void run_gemm_kernel(
    int m, int n, int k, const cutlass::half_t* a, int64_t lda, const cutlass::half_t* b, int64_t ldb,
    cutlass::half_t* c, int64_t ldc, float alpha, float beta, cudaStream_t stream) {

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<T_M, T_N, T_K>,
        cutlass::gemm::GemmShape<W_M, W_N, W_K>,
        cutlass::gemm::GemmShape<I_M, I_N, I_K>,
        cutlass::epilogue::thread::LinearCombination<
            cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value,
            float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages>;

    typename Gemm::Arguments args({m, n, k}, {a, lda}, {b, ldb}, {c, ldc}, {c, ldc}, {alpha, beta});
    Gemm gemm_op;
    if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) 
        throw std::runtime_error("Kernel cannot be implemented");
    if (gemm_op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("Kernel initialization failed");
    if (gemm_op(stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("Kernel execution failed");
}
"""

    # --- 生成所有包装函数 ---
    for _, row in best_configs.iterrows():
        m, n, k = int(row['m']), int(row['n']), int(row['k'])
        cta_m, cta_n, cta_k = int(row['cta_m']), int(row['cta_n']), int(row['cta_k'])
        warps_m, warps_n, warps_k = int(row['warps_m']), int(row['warps_n']), int(row['warps_k'])
        inst_m, inst_n, inst_k = int(row['inst_m']), int(row['inst_n']), int(row['inst_k'])
        stages = int(row['stages'])
        warp_m, warp_n, warp_k = cta_m // warps_m, cta_n // warps_n, cta_k // warps_k
        
        wrapper_name = f"gemm_kernel_wrapper_{m}_{n}_{k}"
        cu_content += f"""
void {wrapper_name}(int m, int n, int k, const cutlass::half_t* a, int64_t lda, const cutlass::half_t* b, int64_t ldb, cutlass::half_t* c, int64_t ldc, float alpha, float beta, cudaStream_t stream) {{
    run_gemm_kernel<{cta_m}, {cta_n}, {cta_k}, {warp_m}, {warp_n}, {warp_k}, {inst_m}, {inst_n}, {inst_k}, {stages}>
    (m, n, k, a, lda, b, ldb, c, ldc, alpha, beta, stream);
}}
"""

    # --- 生成调度器和注册表 (和之前一样，但在命名空间内) ---
    cu_content += """
class GemmKernelRegistry {
public:
    struct ProblemSize { int m, n, k; bool operator<(const ProblemSize& o) const { return std::tie(m, n, k) < std::tie(o.m, o.n, o.k); } };
    static GemmKernelRegistry& instance() { static GemmKernelRegistry reg; return reg; }
    GemmFuncPtr find_kernel(const ProblemSize& size, bool& exact_match) {
        auto it = dispatch_table.find(size);
        if (it != dispatch_table.end()) { exact_match = true; return it->second; }
        exact_match = false; return find_nearest_kernel(size);
    }
private:
    GemmKernelRegistry() {
"""
    for idx, row in best_configs.iterrows():
        m, n, k = int(row['m']), int(row['n']), int(row['k'])
        wrapper_name = f"gemm_kernel_wrapper_{m}_{n}_{k}"
        cu_content += f"        dispatch_table[{{{m}, {n}, {k}}}] = &{wrapper_name};\n"

    cu_content += """
    }
    GemmFuncPtr find_nearest_kernel(const ProblemSize& target) const {
        double min_dist_sq = std::numeric_limits<double>::max(); ProblemSize best_match = {0,0,0};
        for (const auto& pair : dispatch_table) {
            long long dm = (long long)target.m-pair.first.m, dn = (long long)target.n-pair.first.n, dk = (long long)target.k-pair.first.k;
            double dist_sq = (double)dm*dm + (double)dn*dn + (double)dk*dk;
            if (dist_sq < min_dist_sq) { min_dist_sq = dist_sq; best_match = pair.first; }
        }
        return dispatch_table.at(best_match);
    }
    std::map<ProblemSize, GemmFuncPtr> dispatch_table;
};

} // namespace my_cutlass_gemm_impl

// --- 实现暴露给外部的 C-style 接口函数 ---
//
extern "C" void myCutlassHgemm(
    int m, int n, int k,
    const cutlass::half_t* a, long long int lda,
    const cutlass::half_t* b, long long int ldb,
    cutlass::half_t* c, long long int ldc,
    float alpha, float beta, cudaStream_t stream
) {
    using namespace my_cutlass_gemm_impl;
    GemmKernelRegistry::ProblemSize problem = {m, n, k};
    bool exact_match = false;
    GemmFuncPtr kernel_to_run = GemmKernelRegistry::instance().find_kernel(problem, exact_match);
    if (kernel_to_run) {
        kernel_to_run(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta, stream);
    } else {
        // 或者可以打印错误信息后返回一个错误码
        throw std::runtime_error("No optimal CUTLASS kernel could be found.");
    }
}
"""
    with open(cu_path, 'w') as f:
        f.write(cu_content)
    print(f"成功生成实现源文件: '{cu_path}'")


if __name__ == "__main__":
    CSV_FILE = 'result.csv'
    generate_so_and_header(CSV_FILE)