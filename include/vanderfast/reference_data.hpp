// Immutable D3 reference tables (format v1, see docs/data-provenance.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vanderfast {

class ReferenceData {
 public:
  static constexpr int kMaxRef = 7;

  // The tables embedded at build time (simple-dftd3 v1.6.0).
  static std::shared_ptr<const ReferenceData> embedded();

  // Validate and copy a table blob. ``identity`` is a caller-supplied label
  // (for example its sha256) reported in diagnostics.
  static std::shared_ptr<const ReferenceData> from_bytes(const void* data, std::size_t size,
                                                         std::string identity);

  int max_z() const noexcept { return max_z_; }
  int max_ref() const noexcept { return max_ref_; }
  bool supports(int z) const noexcept { return z >= 1 && z <= max_z_; }
  int nref(int z) const;
  double reference_cn(int z, int ref) const;
  // Reference C6 with ``ref_i`` belonging to ``zi`` (upstream get_c6(ref_i, ref_j, zi, zj)).
  double c6(int zi, int zj, int ref_i, int ref_j) const;
  double rcov(int z) const;
  double r4r2(int z) const;
  double rvdw(int zi, int zj) const;

  const std::string& source_id() const noexcept { return source_id_; }
  const std::string& identity() const noexcept { return identity_; }

  ReferenceData(const ReferenceData&) = delete;
  ReferenceData& operator=(const ReferenceData&) = delete;
  ~ReferenceData();

 private:
  ReferenceData() = default;
  void check(int z) const;
  static std::size_t pair_index(int zi, int zj) noexcept;

  int max_z_ = 0;
  int max_ref_ = 0;
  std::vector<std::uint8_t> nref_;
  std::vector<std::size_t> ref_offset_;
  std::vector<double> ref_cn_;
  std::vector<double> rcov_;
  std::vector<double> r4r2_;
  std::vector<double> c6_;
  std::vector<std::size_t> c6_offset_;
  std::vector<double> rvdw_;
  std::string source_id_;
  std::string identity_;
};

}  // namespace vanderfast
