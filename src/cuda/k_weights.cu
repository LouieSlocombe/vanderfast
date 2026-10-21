// K2: Gaussian reference weights and their CN derivatives (upstream model.f90).
#include "cuda_check.cuh"
#include "device_util.cuh"
#include "kernels.cuh"

namespace vanderfast::cuda {

namespace {

template <bool Derivatives>
__global__ void weights_kernel(ModelView m, GeometryView g, const double* __restrict__ cn,
                               double* __restrict__ w, double* __restrict__ dw) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= g.n) return;
  const int s = species(g, i);
  const int nref = m.nref[s];
  const double* cnref = m.cnref + s * kRefPad;
  const unsigned maxmask = m.maxmask[s];
  const double c = cn[i];
  const double wf2 = 2.0 * math::kWf;
  double gauss[kRefPad];
  double norm = 0.0;
  double dnorm = 0.0;
  for (int a = 0; a < kRefPad; ++a) {
    gauss[a] = 0.0;
    if (a >= nref) continue;
    const double delta = c - cnref[a];
    gauss[a] = exp(-math::kWf * (delta * delta));
    norm += gauss[a];
    if (Derivatives) dnorm += wf2 * (cnref[a] - c) * gauss[a];
  }
  norm = 1.0 / norm;
  for (int a = 0; a < kRefPad; ++a) {
    double value = 0.0;
    double derivative = 0.0;
    if (a < nref) {
      value = gauss[a] * norm;
      if (math::is_exceptional(value)) value = ((maxmask >> a) & 1u) ? 1.0 : 0.0;
      if (Derivatives) {
        const double expd = wf2 * (cnref[a] - c) * gauss[a];
        derivative = expd * norm - gauss[a] * dnorm * (norm * norm);
        if (math::is_exceptional(derivative)) derivative = 0.0;
      }
    }
    w[i * kRefPad + a] = value;
    if (Derivatives) dw[i * kRefPad + a] = derivative;
  }
}

}  // namespace

void launch_weights(const ModelView& m, const GeometryView& g, const double* cn, double* w,
                    double* dw, cudaStream_t stream) {
  if (g.n == 0) return;
  constexpr int threads = 128;
  const int blocks = (g.n + threads - 1) / threads;
  if (dw != nullptr) {
    weights_kernel<true><<<blocks, threads, 0, stream>>>(m, g, cn, w, dw);
  } else {
    weights_kernel<false><<<blocks, threads, 0, stream>>>(m, g, cn, w, dw);
  }
  VF_LAUNCH_CHECK(stream);
}

}  // namespace vanderfast::cuda
