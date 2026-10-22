#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>

#include "core/d4_model.hpp"
#include "core/ewald.hpp"
#include "vanderfast/build_info.hpp"
#include "vanderfast/d4_parameters.hpp"
#include "vanderfast/d4_reference_data.hpp"
#include "vanderfast/error.hpp"

using vanderfast::D4Parameters;
using vanderfast::D4ReferenceData;
using vanderfast::EeqOptions;
using vanderfast::Error;

namespace {

std::vector<unsigned char> blob() {
  std::ifstream file(VANDERFAST_D4_REFERENCE_BLOB_PATH, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// A skewed cell (rows are lattice vectors, bohr); the expected values below
// come from the NumPy oracle (vanderfast._oracle_d4).
constexpr std::array<double, 9> kSkewed{7.1, 0.0, 0.0, 1.3, 6.4, 0.0, -0.8, 0.9, 8.2};

}  // namespace

TEST_CASE("embedded D4 tables match upstream spot values") {
  auto tables = D4ReferenceData::embedded();
  CHECK(tables->max_z() == 103);
  CHECK(tables->n_secondary() == 17);
  CHECK(tables->identity() == VANDERFAST_D4_REFERENCE_SHA256);
  CHECK(tables->source_id().rfind("dftd4 v4.2.0", 0) == 0);
  CHECK(tables->nref(1) == 2);
  CHECK(tables->nref(6) == 7);
  CHECK(tables->nref(92) == 7);  // includes upstream's zero-filled phantom reference
  const std::size_t carbon = tables->first_ref(6);
  const int ngw[] = {3, 3, 1, 3, 1, 3, 3};
  for (int r = 0; r < 7; ++r) CHECK(tables->ngw(carbon + static_cast<std::size_t>(r)) == ngw[r]);
  CHECK(tables->ref_value(carbon + 1, D4ReferenceData::kRefCn) == 0.98678926403687);
  CHECK(tables->element(1, D4ReferenceData::kEeqChi) == 1.23695041);
  CHECK(tables->element(1, D4ReferenceData::kEeqEta) == -0.35015861);
  CHECK(tables->alphaiw(tables->first_ref(1))[0] == 5.054016);
  CHECK(tables->frequencies()[0] == 1e-6);
  CHECK(tables->weights()[0] == 0.0249995);
}

TEST_CASE("D4 accessors reject unsupported elements") {
  auto tables = D4ReferenceData::embedded();
  CHECK_FALSE(tables->supports(0));
  CHECK_FALSE(tables->supports(104));
  CHECK_THROWS_AS(tables->nref(104), Error);
  CHECK_THROWS_AS(tables->first_ref(0), Error);
  CHECK_THROWS_AS(tables->element(104, D4ReferenceData::kZeff), Error);
}

TEST_CASE("D4 from_bytes validates the structure") {
  std::vector<unsigned char> data = blob();
  REQUIRE(data.size() > 512);
  auto copy = D4ReferenceData::from_bytes(data.data(), data.size(), "copy");
  CHECK(copy->identity() == "copy");
  CHECK(copy->element(26, D4ReferenceData::kR4r2) ==
        D4ReferenceData::embedded()->element(26, D4ReferenceData::kR4r2));

  CHECK_THROWS_AS(D4ReferenceData::from_bytes(data.data(), 100, ""), Error);
  auto corrupt = [&](std::size_t offset, unsigned char value) {
    std::vector<unsigned char> bad = data;
    bad[offset] = value;
    return D4ReferenceData::from_bytes(bad.data(), bad.size(), "");
  };
  CHECK_THROWS_AS(corrupt(0, 'X'), Error);    // magic
  CHECK_THROWS_AS(corrupt(8, 2), Error);      // version
  CHECK_THROWS_AS(corrupt(20, 0), Error);     // flags
  CHECK_THROWS_AS(corrupt(28, 6), Error);     // max_ref
  CHECK_THROWS_AS(corrupt(112, 5), Error);    // section count
  CHECK_THROWS_AS(corrupt(124, 9), Error);    // dtype
  CHECK_THROWS_AS(corrupt(128, 1), Error);    // misaligned offset
  std::uint64_t nref_offset = 0;
  std::memcpy(&nref_offset, data.data() + 128, sizeof(nref_offset));
  CHECK_THROWS_AS(corrupt(nref_offset + 1, 9), Error);  // reference count out of range
  CHECK_THROWS_AS(corrupt(nref_offset, 1), Error);      // slot 0 must be zero

  // A reference CN that moves to another histogram bin breaks the stored
  // Gaussian counts.
  std::uint64_t scalars = 0;
  std::memcpy(&scalars, data.data() + 120 + 3 * 24 + 8, sizeof(scalars));
  std::vector<unsigned char> moved = data;
  const std::size_t carbon_ref1 = D4ReferenceData::embedded()->first_ref(6) + 1;
  const std::size_t refcn = scalars + 8 * (carbon_ref1 * D4ReferenceData::kRefColumns + D4ReferenceData::kRefCn);
  const double shifted = 5.0;
  std::memcpy(moved.data() + refcn, &shifted, sizeof(shifted));
  CHECK_THROWS_AS(D4ReferenceData::from_bytes(moved.data(), moved.size(), ""), Error);
}

TEST_CASE("D4 parameter validation and activity rules") {
  D4Parameters p;
  p.s8 = 0.95948085;
  p.a1 = 0.38574991;
  p.a2 = 4.80688534;
  CHECK_NOTHROW(p.validate());
  CHECK(p.pair_active());
  CHECK(p.atm_active());
  CHECK(p.atm_exponent() == 16.0);
  CHECK(p.cn_radius() == 30.0);
  p.cutoffs.cn = 20.0;
  CHECK(p.cn_radius() == D4Parameters::kEeqCnCutoff);
  const auto fingerprint = p.fingerprint();
  p.ga = 2.0;
  CHECK(p.fingerprint() != fingerprint);
  p.s9 = 0.0;
  CHECK_FALSE(p.atm_active());
  p.s6 = 0.0;
  p.s8 = 0.0;
  CHECK_FALSE(p.pair_active());
  p.wf = 0.0;
  CHECK_THROWS_AS(p.validate(), Error);
  p.wf = 6.0;
  p.gc = std::numeric_limits<double>::infinity();
  CHECK_THROWS_AS(p.validate(), Error);
  p.gc = 2.0;
  p.cutoffs.disp2 = 0.0;
  CHECK_THROWS_AS(p.validate(), Error);

  EeqOptions options;
  CHECK_NOTHROW(options.validate());
  options.tolerance = 0.0;
  CHECK_THROWS_AS(options.validate(), Error);
  options.tolerance = 1e-12;
  options.max_iterations = 0;
  CHECK_THROWS_AS(options.validate(), Error);
}

TEST_CASE("D4 charge scaling and species tables") {
  CHECK(vanderfast::d4_zeta(3.0, 2.0, 1.0, -0.5) == std::exp(3.0));
  CHECK(vanderfast::d4_zeta(3.0, 2.0, 1.0, 1.0) == 1.0);
  auto tables = D4ReferenceData::embedded();
  const auto species = vanderfast::make_d4_species(*tables, {1, 6, 92}, 3.0, 2.0);
  CHECK(species.nref == std::vector<int>{2, 7, 7});
  CHECK(species.max_nref == 7);
  CHECK(species.c6ref.size() == 3u * 3u * 64u);
  // Symmetric blocks: C6(H ref 1, C ref 2) == C6(C ref 2, H ref 1).
  const auto block = [&](int s, int t, int a, int b) {
    return species.c6ref[((static_cast<std::size_t>(s) * 3 + static_cast<std::size_t>(t)) * 8 +
                          static_cast<std::size_t>(a)) * 8 + static_cast<std::size_t>(b)];
  };
  CHECK(block(0, 1, 1, 2) == block(1, 0, 2, 1));
  CHECK(block(0, 0, 0, 0) > 0.0);
  // The phantom uranium reference (upstream's third, index 2) has no polarizability and no C6.
  CHECK(block(2, 2, 2, 2) == 0.0);
  CHECK(block(2, 2, 3, 3) > 0.0);
  CHECK_THROWS_AS(vanderfast::make_d4_species(*tables, {104}, 3.0, 2.0), Error);
}

TEST_CASE("upstream Ewald setup replicates multicharge") {
  CHECK(vanderfast::upstream_alpha(kSkewed) == Catch::Approx(0.4194304).epsilon(1e-15));
  const std::array<double, 9> cubic{9.0, 0, 0, 0, 9.0, 0, 0, 0, 9.0};
  CHECK(vanderfast::upstream_alpha(cubic) == Catch::Approx(0.29360127999999996).epsilon(1e-15));

  const auto setup = vanderfast::upstream_ewald(kSkewed);
  CHECK(setup.n_trans() == 125);
  CHECK(setup.n_recip() == 124);
  CHECK(setup.ws_candidates.size() == 27u * 3u);
  const double vec[3] = {1.7, -0.6, 2.3};
  CHECK(vanderfast::ewald_value(setup, vec, 0.83) == Catch::Approx(0.030028503295088646).epsilon(1e-13));
  double dg[3];
  double ds[9];
  vanderfast::ewald_derivative(setup, vec, 0.83, dg, ds);
  const double dg_ref[3] = {-0.05074645795913252, 0.019637360594396525, -0.08013309644736762};
  const double ds_ref[9] = {-0.05323608172259762, 0.029546646637053017, -0.10927898927647357,
                            0.029546646637053017, 0.05530765450218495,  0.03831166869104418,
                            -0.10927898927647357, 0.03831166869104418,  -0.12533818493335638};
  for (int k = 0; k < 3; ++k) CHECK(dg[k] == Catch::Approx(dg_ref[k]).epsilon(1e-12));
  for (int k = 0; k < 9; ++k) CHECK(ds[k] == Catch::Approx(ds_ref[k]).epsilon(1e-12));

  std::vector<double> positions{0.3, -0.2, 0.1, 2.0, 1.0, 9.5};
  vanderfast::wrap_to_central_cell(positions, kSkewed);
  const double wrapped[6] = {1.6, 6.2, 0.1, 4.1, 6.5, 1.3};
  for (int k = 0; k < 6; ++k) CHECK(positions[static_cast<std::size_t>(k)] == Catch::Approx(wrapped[k]).epsilon(1e-14));
}

TEST_CASE("mctc 3x3 helpers") {
  const std::array<double, 9> a{2, 1, 0, 0, 3, 1, 1, 0, 4};
  const double det = vanderfast::mctc_det(a);
  CHECK(det == Catch::Approx(25.0));
  const auto inv = vanderfast::mctc_inverse(a);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k) sum += a[static_cast<std::size_t>(i + 3 * k)] * inv[static_cast<std::size_t>(k + 3 * j)];
      CHECK(sum == Catch::Approx(i == j ? 1.0 : 0.0).margin(1e-15));
    }
  }
}
