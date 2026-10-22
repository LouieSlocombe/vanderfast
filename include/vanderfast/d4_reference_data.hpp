// Immutable DFT-D4 reference tables (format VDFD4TBL v1, see docs/data-provenance.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vanderfast {

class D4ReferenceData {
 public:
  static constexpr int kMaxRef = 7;
  static constexpr int kFreq = 23;

  // Per-reference scalar columns.
  enum RefColumn : int { kRefCovCn = 0, kRefCn, kClsq, kClsh, kHcount, kAscale, kRefColumns };
  // Per-element columns.
  enum ElementColumn : int {
    kZeff = 0,
    kHardness,
    kEn,
    kRcovD4,
    kR4r2,
    kEeqChi,
    kEeqEta,
    kEeqKcnchi,
    kEeqRad,
    kEeqRcov,
    kElementColumns
  };

  // The tables embedded at build time (dftd4 v4.2.0 with multicharge v0.5.0).
  static std::shared_ptr<const D4ReferenceData> embedded();

  // Validate and copy a table blob; ``identity`` labels it in diagnostics.
  static std::shared_ptr<const D4ReferenceData> from_bytes(const void* data, std::size_t size,
                                                           std::string identity);

  int max_z() const noexcept { return max_z_; }
  int n_secondary() const noexcept { return n_secondary_; }
  bool supports(int z) const noexcept { return z >= 1 && z <= max_z_; }
  int nref(int z) const;
  // Index of the first reference of ``z`` in the per-reference arrays.
  std::size_t first_ref(int z) const;
  int ngw(std::size_t ref) const { return ngw_.at(ref); }
  int refsys(std::size_t ref) const { return refsys_.at(ref); }
  double ref_value(std::size_t ref, RefColumn column) const {
    return ref_scalars_.at(ref * kRefColumns + static_cast<std::size_t>(column));
  }
  const double* alphaiw(std::size_t ref) const { return alphaiw_.data() + ref * kFreq; }
  double element(int z, ElementColumn column) const;
  double sscale(int system) const { return secondary_.at(static_cast<std::size_t>(system) * (kFreq + 1)); }
  const double* secaiw(int system) const {
    return secondary_.data() + static_cast<std::size_t>(system) * (kFreq + 1) + 1;
  }
  const double* frequencies() const noexcept { return freq_.data(); }
  const double* weights() const noexcept { return freq_.data() + kFreq; }

  const std::string& source_id() const noexcept { return source_id_; }
  const std::string& identity() const noexcept { return identity_; }

  D4ReferenceData(const D4ReferenceData&) = delete;
  D4ReferenceData& operator=(const D4ReferenceData&) = delete;
  ~D4ReferenceData();

 private:
  D4ReferenceData() = default;
  void check(int z) const;

  int max_z_ = 0;
  int n_secondary_ = 0;
  std::vector<std::uint8_t> nref_;
  std::vector<std::size_t> ref_offset_;
  std::vector<std::uint8_t> ngw_;
  std::vector<std::uint8_t> refsys_;
  std::vector<double> ref_scalars_;
  std::vector<double> alphaiw_;
  std::vector<double> secondary_;
  std::vector<double> element_;
  std::vector<double> freq_;
  std::string source_id_;
  std::string identity_;
};

}  // namespace vanderfast
