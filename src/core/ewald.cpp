#include "ewald.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>

#include "vanderfast/error.hpp"

namespace vanderfast {

namespace {

constexpr double kPi = 3.141592653589793;
const double kSqrtPi = std::sqrt(kPi);
const double kEps = std::sqrt(DBL_EPSILON);  // multicharge eps = sqrt(epsilon)
constexpr double kWsTol = 0.01;              // Wigner-Seitz tie tolerance (bohr^2)
constexpr double kTail = 1e-20;              // converged-mode truncation per term
constexpr int kAlphaIterations = 30;

// Fortran-ordered 3x3: element (i, j) at i + 3 j.
using Mat = std::array<double, 9>;
inline double at(const Mat& a, int i, int j) { return a[static_cast<std::size_t>(i + 3 * j)]; }

// Fortran lattice (columns are vectors) from user cell rows.
Mat lattice_from_cell(const std::array<double, 9>& cell) {
  Mat lattice{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) lattice[static_cast<std::size_t>(c + 3 * r)] = cell[static_cast<std::size_t>(3 * r + c)];
  }
  return lattice;
}

// 2 pi transpose(inverse(lattice)): columns are the reciprocal vectors.
Mat reciprocal(const Mat& lattice) {
  const Mat inv = mctc_inverse(lattice);
  Mat rec{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) rec[static_cast<std::size_t>(i + 3 * j)] = 2.0 * kPi * at(inv, j, i);
  }
  return rec;
}

// mctc get_lattice_points (rep form) in its loop order.
std::vector<double> lattice_points(const Mat& lat, const int rep[3], bool origin) {
  std::vector<double> points;
  for (int ix = 0; ix <= rep[0]; ++ix) {
    for (int iy = 0; iy <= rep[1]; ++iy) {
      for (int iz = 0; iz <= rep[2]; ++iz) {
        if (!origin && ix == 0 && iy == 0 && iz == 0) continue;
        for (int jx = 1; jx >= (ix > 0 ? -1 : 1); jx -= 2) {
          for (int jy = 1; jy >= (iy > 0 ? -1 : 1); jy -= 2) {
            for (int jz = 1; jz >= (iz > 0 ? -1 : 1); jz -= 2) {
              for (int k = 0; k < 3; ++k) {
                points.push_back((at(lat, k, 0) * ix * jx + at(lat, k, 1) * iy * jy) + at(lat, k, 2) * iz * jz);
              }
            }
          }
        }
      }
    }
  }
  return points;
}

double rec_dir_diff(double alpha, double rlen, double dlen, double volume) {
  auto rec = [&](double gg) {
    return 4.0 * kPi * (std::exp(-0.25 * gg * gg / (alpha * alpha)) / (volume * gg * gg));
  };
  auto dir = [&](double rr) { return std::erfc(alpha * rr) / rr; };
  return (rec(4 * rlen) - rec(5 * rlen)) - (dir(2 * dlen) - dir(3 * dlen));
}

double column_norm_min(const Mat& m) {
  double best = 0.0;
  for (int j = 0; j < 3; ++j) {
    const double sum = at(m, 0, j) * at(m, 0, j) + at(m, 1, j) * at(m, 1, j) + at(m, 2, j) * at(m, 2, j);
    best = j == 0 ? sum : std::min(best, sum);
  }
  return std::sqrt(best);
}

// Every lattice point within ``radius`` of the origin (columns of ``lat`` are vectors).
std::vector<double> sphere_points(const Mat& lat, double radius, bool origin) {
  double normals[3][3];
  for (int k = 0; k < 3; ++k) {
    const int a = (k + 1) % 3;
    const int b = (k + 2) % 3;
    normals[k][0] = at(lat, 1, a) * at(lat, 2, b) - at(lat, 2, a) * at(lat, 1, b);
    normals[k][1] = at(lat, 2, a) * at(lat, 0, b) - at(lat, 0, a) * at(lat, 2, b);
    normals[k][2] = at(lat, 0, a) * at(lat, 1, b) - at(lat, 1, a) * at(lat, 0, b);
  }
  const double volume = std::fabs(mctc_det(lat));
  int rep[3];
  for (int k = 0; k < 3; ++k) {
    const double area = std::sqrt(normals[k][0] * normals[k][0] + normals[k][1] * normals[k][1] +
                                  normals[k][2] * normals[k][2]);
    rep[k] = static_cast<int>(std::ceil(radius / (volume / area)));
  }
  std::vector<double> points;
  for (int i = -rep[0]; i <= rep[0]; ++i) {
    for (int j = -rep[1]; j <= rep[1]; ++j) {
      for (int k = -rep[2]; k <= rep[2]; ++k) {
        if (!origin && i == 0 && j == 0 && k == 0) continue;
        double p[3];
        for (int c = 0; c < 3; ++c) p[c] = i * at(lat, c, 0) + j * at(lat, c, 1) + k * at(lat, c, 2);
        if (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= radius * radius) points.insert(points.end(), p, p + 3);
      }
    }
  }
  return points;
}

}  // namespace

double mctc_det(const std::array<double, 9>& a) noexcept {
  return at(a, 0, 0) * at(a, 1, 1) * at(a, 2, 2) - at(a, 0, 0) * at(a, 1, 2) * at(a, 2, 1) -
         at(a, 0, 1) * at(a, 1, 0) * at(a, 2, 2) + at(a, 0, 1) * at(a, 1, 2) * at(a, 2, 0) +
         at(a, 0, 2) * at(a, 1, 0) * at(a, 2, 1) - at(a, 0, 2) * at(a, 1, 1) * at(a, 2, 0);
}

std::array<double, 9> mctc_inverse(const std::array<double, 9>& a) noexcept {
  const double d = 1.0 / mctc_det(a);
  Mat b{};
  auto set = [&](int i, int j, double v) { b[static_cast<std::size_t>(i + 3 * j)] = v; };
  set(0, 0, +d * (at(a, 1, 1) * at(a, 2, 2) - at(a, 1, 2) * at(a, 2, 1)));
  set(1, 0, -d * (at(a, 1, 0) * at(a, 2, 2) - at(a, 1, 2) * at(a, 2, 0)));
  set(2, 0, +d * (at(a, 1, 0) * at(a, 2, 1) - at(a, 1, 1) * at(a, 2, 0)));
  set(0, 1, -d * (at(a, 0, 1) * at(a, 2, 2) - at(a, 0, 2) * at(a, 2, 1)));
  set(1, 1, +d * (at(a, 0, 0) * at(a, 2, 2) - at(a, 0, 2) * at(a, 2, 0)));
  set(2, 1, -d * (at(a, 0, 0) * at(a, 2, 1) - at(a, 0, 1) * at(a, 2, 0)));
  set(0, 2, +d * (at(a, 0, 1) * at(a, 1, 2) - at(a, 0, 2) * at(a, 1, 1)));
  set(1, 2, -d * (at(a, 0, 0) * at(a, 1, 2) - at(a, 0, 2) * at(a, 1, 0)));
  set(2, 2, +d * (at(a, 0, 0) * at(a, 1, 1) - at(a, 0, 1) * at(a, 1, 0)));
  return b;
}

void wrap_to_central_cell(std::vector<double>& positions, const std::array<double, 9>& cell) {
  const Mat lattice = lattice_from_cell(cell);
  const Mat inv = mctc_inverse(lattice);
  constexpr double eps = 1.0e-14;
  for (std::size_t base = 0; base + 2 < positions.size(); base += 3) {
    const double* x = positions.data() + base;
    double frac[3];
    for (int r = 0; r < 3; ++r) frac[r] = (at(inv, r, 0) * x[0] + at(inv, r, 1) * x[1]) + at(inv, r, 2) * x[2];
    for (double& f : frac) {
      const double in = f;
      double out = in;
      if (in < (0.0 - eps)) out = in + std::ceil(-in);
      if (in > (1.0 + eps)) out = in - std::floor(in);
      if (std::fabs(in - 1.0) < eps) out = in - 1.0;
      f = out;
    }
    for (int r = 0; r < 3; ++r) {
      positions[base + static_cast<std::size_t>(r)] =
          (at(lattice, r, 0) * frac[0] + at(lattice, r, 1) * frac[1]) + at(lattice, r, 2) * frac[2];
    }
  }
}

double upstream_alpha(const std::array<double, 9>& cell) {
  const Mat lattice = lattice_from_cell(cell);
  const double volume = std::fabs(mctc_det(lattice));
  const Mat rec = reciprocal(lattice);
  const double rlen = column_norm_min(rec);
  const double dlen = column_norm_min(lattice);
  const double tol = kEps;
  constexpr double alpha0 = 1.0e-8;
  double alpha = alpha0;
  double diff = rec_dir_diff(alpha, rlen, dlen, volume);
  while (diff < -tol && alpha <= DBL_MAX) {
    alpha *= 2.0;
    diff = rec_dir_diff(alpha, rlen, dlen, volume);
  }
  if (alpha > DBL_MAX || alpha == alpha0) return 0.25;
  double left = 0.5 * alpha;
  while (diff < tol && alpha <= DBL_MAX) {
    alpha *= 2.0;
    diff = rec_dir_diff(alpha, rlen, dlen, volume);
  }
  if (alpha > DBL_MAX) return 0.25;
  double right = alpha;
  alpha = (left + right) * 0.5;
  int iterations = 0;
  diff = rec_dir_diff(alpha, rlen, dlen, volume);
  while (std::fabs(diff) > tol && iterations <= kAlphaIterations) {
    if (diff < 0) {
      left = alpha;
    } else {
      right = alpha;
    }
    alpha = (left + right) * 0.5;
    diff = rec_dir_diff(alpha, rlen, dlen, volume);
    iterations += 1;
  }
  return iterations > kAlphaIterations ? 0.25 : alpha;
}

EwaldSetup upstream_ewald(const std::array<double, 9>& cell) {
  const Mat lattice = lattice_from_cell(cell);
  EwaldSetup setup;
  setup.mode = EwaldMode::Upstream;
  setup.alpha = upstream_alpha(cell);
  setup.volume = std::fabs(mctc_det(lattice));
  const int two[3] = {2, 2, 2};
  const int one[3] = {1, 1, 1};
  setup.dtrans = lattice_points(lattice, two, true);
  setup.rtrans = lattice_points(reciprocal(lattice), two, false);
  setup.ws_candidates = lattice_points(lattice, one, true);
  // Self images: the origin is dropped from the distance list, but the kept
  // positions index the full candidate list (multicharge get_pairs).
  std::vector<double> dist;
  double best = DBL_MAX;
  for (std::size_t c = 0; c < 27; ++c) {
    const double* t = setup.ws_candidates.data() + 3 * c;
    const double r2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
    if (r2 < kEps) continue;
    dist.push_back(r2);
    best = std::min(best, r2);
  }
  if (dist.empty()) fail(ErrorCode::InvalidGeometry, "degenerate lattice: every lattice vector is zero");
  std::vector<std::size_t> order(dist.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return dist[a] < dist[b]; });
  for (std::size_t p : order) {
    if (std::fabs(dist[p] - best) > kWsTol) break;
    setup.self_images.insert(setup.self_images.end(), setup.ws_candidates.begin() + static_cast<std::ptrdiff_t>(3 * p),
                             setup.ws_candidates.begin() + static_cast<std::ptrdiff_t>(3 * p + 3));
  }
  return setup;
}

EwaldSetup converged_ewald(const std::array<double, 9>& cell, double alpha, double gam_min, double span) {
  const Mat lattice = lattice_from_cell(cell);
  EwaldSetup setup;
  setup.mode = EwaldMode::Converged;
  setup.volume = std::fabs(mctc_det(lattice));
  setup.alpha = alpha > 0.0 ? alpha : kSqrtPi / std::cbrt(setup.volume);
  const double slowest = std::min(setup.alpha, gam_min);
  double rc = 1.0;
  while (std::erfc(slowest * rc) / rc > kTail) rc *= 1.1;
  setup.real_cutoff = rc;
  setup.dtrans = sphere_points(lattice, rc + span, true);
  double gc = 1.0;
  while ((4 * kPi / setup.volume) * std::exp(-0.25 * gc * gc / (setup.alpha * setup.alpha)) / (gc * gc) > kTail) {
    gc *= 1.1;
  }
  setup.rtrans = sphere_points(reciprocal(lattice), gc, false);
  return setup;
}

double ewald_value(const EwaldSetup& setup, const double vec[3], double gam) {
  const double alpha = setup.alpha;
  double real = 0.0;
  for (std::size_t t = 0; t < setup.n_trans(); ++t) {
    const double* tr = setup.dtrans.data() + 3 * t;
    const double v[3] = {vec[0] + tr[0], vec[1] + tr[1], vec[2] + tr[2]};
    const double r1 = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (r1 < kEps) continue;
    if (setup.mode == EwaldMode::Converged) {
      if (r1 > setup.real_cutoff) continue;
      real += (std::erfc(alpha * r1) - std::erfc(gam * r1)) / r1;
    } else {
      real += std::erf(gam * r1) / r1 - std::erf(alpha * r1) / r1;
    }
  }
  double rec = 0.0;
  const double fac = 4 * kPi / setup.volume;
  for (std::size_t g = 0; g < setup.n_recip(); ++g) {
    const double* G = setup.rtrans.data() + 3 * g;
    const double g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
    const double gv = vec[0] * G[0] + vec[1] * G[1] + vec[2] * G[2];
    rec += std::cos(gv) * fac * std::exp(-0.25 * g2 / (alpha * alpha)) / g2;
  }
  return real + rec;
}

void ewald_derivative(const EwaldSetup& setup, const double vec[3], double gam, double dg[3], double ds[9]) {
  const double alpha = setup.alpha;
  const double gam2 = gam * gam;
  const double alp2 = alpha * alpha;
  for (int k = 0; k < 3; ++k) dg[k] = 0.0;
  for (int k = 0; k < 9; ++k) ds[k] = 0.0;
  for (std::size_t t = 0; t < setup.n_trans(); ++t) {
    const double* tr = setup.dtrans.data() + 3 * t;
    const double v[3] = {vec[0] + tr[0], vec[1] + tr[1], vec[2] + tr[2]};
    const double r1 = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (r1 < kEps) continue;
    const double r2 = r1 * r1;
    double dtmp;
    if (setup.mode == EwaldMode::Converged) {
      if (r1 > setup.real_cutoff) continue;
      const double value = (std::erfc(alpha * r1) - std::erfc(gam * r1)) / r1;
      const double slope = (2 * gam * std::exp(-r2 * gam2) - 2 * alpha * std::exp(-r2 * alp2)) / kSqrtPi;
      dtmp = (slope - value) / r2;
    } else {
      const double gtmp = 2 * gam * std::exp(-r2 * gam2) / (kSqrtPi * r2) - std::erf(r1 * gam) / (r2 * r1);
      const double atmp = -2 * alpha * std::exp(-r2 * alp2) / (kSqrtPi * r2) + std::erf(r1 * alpha) / (r2 * r1);
      dtmp = gtmp + atmp;
    }
    for (int a = 0; a < 3; ++a) {
      dg[a] += dtmp * v[a];
      for (int b = 0; b < 3; ++b) ds[3 * a + b] += dtmp * v[a] * v[b];
    }
  }
  const double fac = 4 * kPi / setup.volume;
  for (std::size_t g = 0; g < setup.n_recip(); ++g) {
    const double* G = setup.rtrans.data() + 3 * g;
    const double g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
    const double gv = vec[0] * G[0] + vec[1] * G[1] + vec[2] * G[2];
    const double etmp = fac * std::exp(-0.25 * g2 / alp2) / g2;
    const double sine = -std::sin(gv) * etmp;
    const double cosine = etmp * std::cos(gv);
    const double gfac = 2.0 / g2 + 0.5 / alp2;
    for (int a = 0; a < 3; ++a) {
      dg[a] += sine * G[a];
      for (int b = 0; b < 3; ++b) ds[3 * a + b] += cosine * (gfac * G[a] * G[b] - (a == b ? 1.0 : 0.0));
    }
  }
}

double ewald_self_value(const EwaldSetup& setup, double gam) {
  if (setup.mode == EwaldMode::Converged) {
    const double zero[3] = {0.0, 0.0, 0.0};
    return ewald_value(setup, zero, gam);
  }
  const std::size_t images = setup.self_images.size() / 3;
  const double wsw = 1.0 / static_cast<double>(images);
  double total = 0.0;
  for (std::size_t i = 0; i < images; ++i) total += ewald_value(setup, setup.self_images.data() + 3 * i, gam) * wsw;
  return total;
}

void ewald_self_derivative(const EwaldSetup& setup, double gam, double ds[9]) {
  double dg[3];
  if (setup.mode == EwaldMode::Converged) {
    const double zero[3] = {0.0, 0.0, 0.0};
    ewald_derivative(setup, zero, gam, dg, ds);
    return;
  }
  const std::size_t images = setup.self_images.size() / 3;
  const double wsw = 1.0 / static_cast<double>(images);
  for (int k = 0; k < 9; ++k) ds[k] = 0.0;
  double part[9];
  for (std::size_t i = 0; i < images; ++i) {
    ewald_derivative(setup, setup.self_images.data() + 3 * i, gam, dg, part);
    for (int k = 0; k < 9; ++k) ds[k] += part[k] * wsw;
  }
}

}  // namespace vanderfast
