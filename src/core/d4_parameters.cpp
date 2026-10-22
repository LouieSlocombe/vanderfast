#include "vanderfast/d4_parameters.hpp"

#include <cfloat>
#include <cmath>
#include <cstring>

#include "vanderfast/error.hpp"

namespace vanderfast {

void D4Parameters::validate() const {
  const double values[] = {s6, s8, a1, a2, s9, alp, ga, gc, wf, cutoffs.cn, cutoffs.disp2, cutoffs.disp3};
  for (double value : values) {
    if (!std::isfinite(value)) fail(ErrorCode::InvalidParameter, "parameters must be finite");
  }
  if (alp <= 0.0) fail(ErrorCode::InvalidParameter, "alp must be positive");
  if (wf <= 0.0) fail(ErrorCode::InvalidParameter, "wf must be positive");
  if (cutoffs.cn <= 0.0 || cutoffs.disp2 <= 0.0 || cutoffs.disp3 <= 0.0) {
    fail(ErrorCode::InvalidParameter, "cutoffs must be positive");
  }
}

bool D4Parameters::pair_active() const noexcept {
  return !(std::fabs(s6) < DBL_EPSILON && std::fabs(s8) < DBL_EPSILON);
}

bool D4Parameters::atm_active() const noexcept { return std::fabs(s9) >= DBL_EPSILON; }

std::uint64_t D4Parameters::fingerprint() const noexcept {
  // FNV-1a over the exact bit patterns of every value.
  const double values[] = {s6, s8, a1, a2, s9, alp, ga, gc, wf, cutoffs.cn, cutoffs.disp2, cutoffs.disp3};
  std::uint64_t hash = 1469598103934665603ull;
  for (double value : values) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= (bits >> (8 * byte)) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

void EeqOptions::validate() const {
  if (!(tolerance > 0.0 && tolerance < 1.0)) {
    fail(ErrorCode::InvalidArgument, "the charge-solver tolerance must be in (0, 1)");
  }
  if (max_iterations < 1) fail(ErrorCode::InvalidArgument, "max_iterations must be positive");
}

}  // namespace vanderfast
