#include "vanderfast/reference_data.hpp"

#include <cmath>
#include <cstring>
#include <string>

#include "vanderfast/build_info.hpp"
#include "vanderfast/error.hpp"

namespace vanderfast {
namespace detail {
extern const unsigned char d3_reference_blob[];
extern const std::size_t d3_reference_blob_size;
}  // namespace detail

namespace {

constexpr char kMagic[8] = {'V', 'D', 'F', 'D', '3', 'T', 'B', 'L'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kHeaderBytes = 512;
constexpr std::uint32_t kByteOrderMark = 0x01020304u;
constexpr std::uint32_t kFlags = (1u << 0) | (1u << 1);
constexpr std::uint32_t kDtypeU8 = 1;
constexpr std::uint32_t kDtypeF64 = 2;
constexpr std::uint64_t kSectionAlign = 64;
constexpr std::size_t kSectionTable = 120;
constexpr std::size_t kSectionEntry = 24;

enum Section : std::uint32_t { kNref = 1, kRefCn, kRcov, kR4r2, kC6, kRvdw };

[[noreturn]] void corrupt(const std::string& what) {
  fail(ErrorCode::InvalidArgument, "reference data is invalid: " + what);
}

template <class T>
T read(const unsigned char* data, std::size_t offset) {
  T value;
  std::memcpy(&value, data + offset, sizeof(T));
  return value;
}

}  // namespace

ReferenceData::~ReferenceData() = default;

std::size_t ReferenceData::pair_index(int zi, int zj) noexcept {
  if (zi < zj) std::swap(zi, zj);
  return static_cast<std::size_t>(zi) * static_cast<std::size_t>(zi - 1) / 2u +
         static_cast<std::size_t>(zj - 1);
}

std::shared_ptr<const ReferenceData> ReferenceData::from_bytes(const void* data, std::size_t size,
                                                               std::string identity) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  if (size < kHeaderBytes) corrupt("shorter than its header");
  if (std::memcmp(bytes, kMagic, sizeof(kMagic)) != 0) corrupt("wrong magic number");
  if (read<std::uint32_t>(bytes, 8) != kFormatVersion ||
      read<std::uint32_t>(bytes, 12) != kHeaderBytes ||
      read<std::uint32_t>(bytes, 16) != kByteOrderMark) {
    corrupt("unsupported format version or byte order");
  }
  if (read<std::uint32_t>(bytes, 20) != kFlags) corrupt("unexpected flags");
  const auto max_z = read<std::uint32_t>(bytes, 24);
  const auto max_ref = read<std::uint32_t>(bytes, 28);
  const auto n_refs = read<std::uint32_t>(bytes, 32);
  const auto n_pairs = read<std::uint32_t>(bytes, 36);
  const auto n_c6 = read<std::uint64_t>(bytes, 40);
  if (max_z < 1 || max_z > 1000 || n_pairs != max_z * (max_z + 1) / 2 || max_ref < 1 ||
      max_ref > static_cast<std::uint32_t>(kMaxRef)) {
    corrupt("inconsistent dimensions");
  }
  if (read<std::uint32_t>(bytes, 112) != 6) corrupt("expected six sections");

  std::shared_ptr<ReferenceData> tables(new ReferenceData());
  tables->max_z_ = static_cast<int>(max_z);
  tables->max_ref_ = static_cast<int>(max_ref);
  tables->identity_ = std::move(identity);
  {
    const char* source = reinterpret_cast<const char*>(bytes + 48);
    tables->source_id_.assign(source, strnlen(source, 64));
  }

  const std::uint64_t expected_count[7] = {0, max_z + 1u, n_refs, max_z + 1u, max_z + 1u, n_c6,
                                           n_pairs};
  const std::uint32_t expected_dtype[7] = {0, kDtypeU8, kDtypeF64, kDtypeF64,
                                           kDtypeF64, kDtypeF64, kDtypeF64};
  std::uint64_t offsets[7] = {};
  for (std::uint32_t index = 0; index < 6; ++index) {
    const std::size_t entry = kSectionTable + index * kSectionEntry;
    const auto id = read<std::uint32_t>(bytes, entry);
    const auto dtype = read<std::uint32_t>(bytes, entry + 4);
    const auto offset = read<std::uint64_t>(bytes, entry + 8);
    const auto count = read<std::uint64_t>(bytes, entry + 16);
    if (id != index + 1 || dtype != expected_dtype[id] || count != expected_count[id]) {
      corrupt("unexpected section " + std::to_string(index));
    }
    const std::uint64_t width = dtype == kDtypeU8 ? 1 : 8;
    if (offset % kSectionAlign != 0 || count > (size - offset) / width || offset > size) {
      corrupt("section " + std::to_string(id) + " is misaligned or out of bounds");
    }
    offsets[id] = offset;
  }

  auto load_f64 = [&](std::uint32_t id, std::vector<double>& out) {
    out.resize(static_cast<std::size_t>(expected_count[id]));
    std::memcpy(out.data(), bytes + offsets[id], out.size() * sizeof(double));
  };
  tables->nref_.assign(bytes + offsets[kNref], bytes + offsets[kNref] + max_z + 1);
  load_f64(kRefCn, tables->ref_cn_);
  load_f64(kRcov, tables->rcov_);
  load_f64(kR4r2, tables->r4r2_);
  load_f64(kC6, tables->c6_);
  load_f64(kRvdw, tables->rvdw_);

  if (tables->nref_[0] != 0) corrupt("reference count slot 0 must be zero");
  tables->ref_offset_.assign(max_z + 2u, 0);
  for (std::uint32_t z = 1; z <= max_z; ++z) {
    const auto n = tables->nref_[z];
    if (n < 1 || n > max_ref) corrupt("reference counts out of range");
    tables->ref_offset_[z + 1] = tables->ref_offset_[z] + n;
  }
  if (tables->ref_offset_[max_z + 1] != n_refs) corrupt("reference counts do not sum up");
  tables->c6_offset_.assign(n_pairs + 1u, 0);
  for (int zi = 1; zi <= tables->max_z_; ++zi) {
    for (int zj = 1; zj <= zi; ++zj) {
      const std::size_t ic = pair_index(zi, zj);
      tables->c6_offset_[ic + 1] = static_cast<std::size_t>(tables->nref_[static_cast<std::size_t>(zi)]) *
                                   tables->nref_[static_cast<std::size_t>(zj)];
    }
  }
  for (std::size_t ic = 0; ic < n_pairs; ++ic) tables->c6_offset_[ic + 1] += tables->c6_offset_[ic];
  if (tables->c6_offset_[n_pairs] != n_c6) corrupt("C6 table length does not match");

  auto finite = [](const std::vector<double>& values, std::size_t first) {
    for (std::size_t i = first; i < values.size(); ++i) {
      if (!std::isfinite(values[i])) return false;
    }
    return true;
  };
  if (!finite(tables->ref_cn_, 0) || !finite(tables->rcov_, 1) || !finite(tables->r4r2_, 1) ||
      !finite(tables->c6_, 0) || !finite(tables->rvdw_, 0)) {
    corrupt("non-finite values");
  }
  return tables;
}

std::shared_ptr<const ReferenceData> ReferenceData::embedded() {
  static const std::shared_ptr<const ReferenceData> tables = from_bytes(
      detail::d3_reference_blob, detail::d3_reference_blob_size, VANDERFAST_REFERENCE_SHA256);
  return tables;
}

void ReferenceData::check(int z) const {
  if (!supports(z)) {
    fail(ErrorCode::UnsupportedElement,
         "atomic number " + std::to_string(z) + " is not supported (1.." + std::to_string(max_z_) + ")");
  }
}

int ReferenceData::nref(int z) const {
  check(z);
  return nref_[static_cast<std::size_t>(z)];
}

double ReferenceData::reference_cn(int z, int ref) const {
  if (ref < 0 || ref >= nref(z)) fail(ErrorCode::InvalidArgument, "reference index out of range");
  return ref_cn_[ref_offset_[static_cast<std::size_t>(z)] + static_cast<std::size_t>(ref)];
}

double ReferenceData::c6(int zi, int zj, int ref_i, int ref_j) const {
  if (ref_i < 0 || ref_i >= nref(zi) || ref_j < 0 || ref_j >= nref(zj)) {
    fail(ErrorCode::InvalidArgument, "reference index out of range");
  }
  const std::size_t start = c6_offset_[pair_index(zi, zj)];
  if (zi >= zj) {
    return c6_[start + static_cast<std::size_t>(ref_i * nref(zj) + ref_j)];
  }
  return c6_[start + static_cast<std::size_t>(ref_j * nref(zi) + ref_i)];
}

double ReferenceData::rcov(int z) const {
  check(z);
  return rcov_[static_cast<std::size_t>(z)];
}

double ReferenceData::r4r2(int z) const {
  check(z);
  return r4r2_[static_cast<std::size_t>(z)];
}

double ReferenceData::rvdw(int zi, int zj) const {
  check(zi);
  check(zj);
  return rvdw_[pair_index(zi, zj)];
}

}  // namespace vanderfast
