// K2b: species-contracted reference weights U_i^(t) = w_i^T K^(s_i t) and dU.
//
// With U, a C6 coefficient and its CN derivatives become dot products of
// length M: C6_ij = U_i^(s_j) . w_j, dC6_ij/dCN_i = dU_i^(s_j) . w_j, and
// dC6_ij/dCN_j = U_i^(s_j) . dw_j. This only reassociates the sums.
#include "cuda_check.cuh"
#include "device_util.cuh"
#include "kernels.cuh"

namespace vanderfast::cuda {

namespace {

template <bool Derivatives>
__global__ void contract_kernel(ModelView m, GeometryView g, const double* __restrict__ w,
                                const double* __restrict__ dw, double* __restrict__ u,
                                double* __restrict__ du) {
  const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int S = m.n_species;
  if (index >= static_cast<long long>(g.n) * S) return;
  const int i = static_cast<int>(index / S);
  const int t = static_cast<int>(index % S);
  const int si = species(g, i);
  const int nref_i = m.nref[si];
  const double* block = m.c6ref + (si * S + t) * kC6Block;
  for (int b = 0; b < kRefPad; ++b) {
    double value = 0.0;
    double derivative = 0.0;
    for (int a = 0; a < nref_i; ++a) {
      const double k = block[a * kRefPad + b];
      value += w[i * kRefPad + a] * k;
      if (Derivatives) derivative += dw[i * kRefPad + a] * k;
    }
    u[index * kRefPad + b] = value;
    if (Derivatives) du[index * kRefPad + b] = derivative;
  }
}

}  // namespace

void launch_contract(const ModelView& m, const GeometryView& g, const double* w, const double* dw,
                     double* u, double* du, cudaStream_t stream) {
  const long long total = static_cast<long long>(g.n) * m.n_species;
  if (total == 0) return;
  constexpr int threads = 128;
  const auto blocks = static_cast<unsigned>((total + threads - 1) / threads);
  if (dw != nullptr) {
    contract_kernel<true><<<blocks, threads, 0, stream>>>(m, g, w, dw, u, du);
  } else {
    contract_kernel<false><<<blocks, threads, 0, stream>>>(m, g, w, dw, u, du);
  }
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
