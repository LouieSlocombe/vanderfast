#include "vanderfast/d4_reference_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "vanderfast/build_info.hpp"
#include "vanderfast/error.hpp"

namespace vanderfast {
namespace detail {
extern const unsigned char d4_reference_blob[];
extern const std::size_t d4_reference_blob_size;
}  // namespace detail

namespace {

constexpr char kMagic[8] = {'V', 'D', 'F', 'D', '4', 'T', 'B', 'L'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kHeaderBytes = 512;
constexpr std::uint32_t kByteOrderMark = 0x01020304u;
constexpr std::uint32_t kFlags = (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3);
constexpr std::uint32_t kDtypeU8 = 1;
constexpr std::uint32_t kDtypeF64 = 2;
constexpr std::uint64_t kSectionAlign = 64;
constexpr std::size_t kSectionTable = 120;
constexpr std::size_t kSectionEntry = 24;
constexpr int kMaxCnBin = 19;

// Section ids; prefixed so they cannot be shadowed by D4ReferenceData members
// (D4ReferenceData::kFreq is the frequency count, not the FREQ section id).
enum Section : std::uint32_t {
  kSectionNref = 1,
  kSectionNgw,
  kSectionRefsys,
  kSectionRefScalars,
  kSectionAlphaiw,
  kSectionSecondary,
  kSectionElement,
  kSectionFreq,
  kSections = 8
};

[[noreturn]] void corrupt(const std::string& what) {
  fail(ErrorCode::InvalidArgument, "D4 reference data is invalid: " + what);
}

template <class T>
T read(const unsigned char* data, std::size_t offset) {
  T value;
  std::memcpy(&value, data + offset, sizeof(T));
  return value;
}

// Upstream set_refgw: Gaussians per reference from the reference-CN histogram
// (Fortran nint rounds half away from zero, as std::lround does).
std::vector<int> gaussian_counts(const double* refcn, int nref, std::size_t stride) {
  int cnc[kMaxCnBin + 1] = {1};
  auto bin = [&](int r) {
    return static_cast<int>(std::min<long>(std::lround(refcn[static_cast<std::size_t>(r) * stride]), kMaxCnBin));
  };
  for (int r = 0; r < nref; ++r) cnc[bin(r)] += 1;
  std::vector<int> counts(static_cast<std::size_t>(nref));
  for (int r = 0; r < nref; ++r) {
    const int icn = cnc[bin(r)];
    counts[static_cast<std::size_t>(r)] = icn * (icn + 1) / 2;
  }
  return counts;
}

}  // namespace

D4ReferenceData::~D4ReferenceData() = default;

std::shared_ptr<const D4ReferenceData> D4ReferenceData::from_bytes(const void* data, std::size_t size,
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
  const auto n_freq = read<std::uint32_t>(bytes, 36);
  const auto n_secondary = read<std::uint64_t>(bytes, 40);
  if (max_z < 1 || max_z > 1000 || max_ref != static_cast<std::uint32_t>(kMaxRef) ||
      n_freq != static_cast<std::uint32_t>(kFreq) || n_secondary < 1 || n_secondary > 255) {
    corrupt("inconsistent dimensions");
  }
  if (read<std::uint32_t>(bytes, 112) != kSections) corrupt("expected eight sections");

  std::shared_ptr<D4ReferenceData> tables(new D4ReferenceData());
  tables->max_z_ = static_cast<int>(max_z);
  tables->n_secondary_ = static_cast<int>(n_secondary);
  tables->identity_ = std::move(identity);
  {
    const char* source = reinterpret_cast<const char*>(bytes + 48);
    tables->source_id_.assign(source, strnlen(source, 64));
  }

  const std::uint64_t expected_count[kSections + 1] = {
      0,
      max_z + 1u,
      n_refs,
      n_refs,
      static_cast<std::uint64_t>(n_refs) * kRefColumns,
      static_cast<std::uint64_t>(n_refs) * kFreq,
      (n_secondary + 1) * (kFreq + 1u),
      static_cast<std::uint64_t>(max_z + 1u) * kElementColumns,
      2u * kFreq};
  const std::uint32_t expected_dtype[kSections + 1] = {0,         kDtypeU8,  kDtypeU8,  kDtypeU8, kDtypeF64,
                                                       kDtypeF64, kDtypeF64, kDtypeF64, kDtypeF64};
  std::uint64_t offsets[kSections + 1] = {};
  for (std::uint32_t index = 0; index < kSections; ++index) {
    const std::size_t entry = kSectionTable + index * kSectionEntry;
    const auto id = read<std::uint32_t>(bytes, entry);
    const auto dtype = read<std::uint32_t>(bytes, entry + 4);
    const auto offset = read<std::uint64_t>(bytes, entry + 8);
    const auto count = read<std::uint64_t>(bytes, entry + 16);
    if (id != index + 1 || dtype != expected_dtype[id] || count != expected_count[id]) {
      corrupt("unexpected section " + std::to_string(index));
    }
    const std::uint64_t width = dtype == kDtypeU8 ? 1 : 8;
    if (offset % kSectionAlign != 0 || offset > size || count > (size - offset) / width) {
      corrupt("section " + std::to_string(id) + " is misaligned or out of bounds");
    }
    offsets[id] = offset;
  }

  auto load_u8 = [&](std::uint32_t id, std::vector<std::uint8_t>& out) {
    out.assign(bytes + offsets[id], bytes + offsets[id] + expected_count[id]);
  };
  auto load_f64 = [&](std::uint32_t id, std::vector<double>& out) {
    out.resize(static_cast<std::size_t>(expected_count[id]));
    std::memcpy(out.data(), bytes + offsets[id], out.size() * sizeof(double));
  };
  load_u8(kSectionNref, tables->nref_);
  load_u8(kSectionNgw, tables->ngw_);
  load_u8(kSectionRefsys, tables->refsys_);
  load_f64(kSectionRefScalars, tables->ref_scalars_);
  load_f64(kSectionAlphaiw, tables->alphaiw_);
  load_f64(kSectionSecondary, tables->secondary_);
  load_f64(kSectionElement, tables->element_);
  load_f64(kSectionFreq, tables->freq_);

  if (tables->nref_[0] != 0) corrupt("reference count slot 0 must be zero");
  tables->ref_offset_.assign(max_z + 2u, 0);
  for (std::uint32_t z = 1; z <= max_z; ++z) {
    const auto n = tables->nref_[z];
    if (n < 1 || n > kMaxRef) corrupt("reference counts out of range");
    tables->ref_offset_[z + 1] = tables->ref_offset_[z] + n;
  }
  if (tables->ref_offset_[max_z + 1] != n_refs) corrupt("reference counts do not sum up");

  auto finite = [](const std::vector<double>& values, std::size_t first) {
    return std::all_of(values.begin() + static_cast<std::ptrdiff_t>(first), values.end(),
                       [](double v) { return std::isfinite(v); });
  };
  if (!finite(tables->ref_scalars_, 0) || !finite(tables->alphaiw_, 0) ||
      !finite(tables->secondary_, kFreq + 1u) || !finite(tables->element_, kElementColumns) ||
      !finite(tables->freq_, 0)) {
    corrupt("non-finite values");
  }
  for (std::uint8_t system : tables->refsys_) {
    if (system > n_secondary) corrupt("a reference names an unknown secondary system");
  }
  for (int k = 1; k < kFreq; ++k) {
    if (!(tables->freq_[static_cast<std::size_t>(k)] > tables->freq_[static_cast<std::size_t>(k - 1)])) {
      corrupt("frequency grid is not increasing");
    }
  }
  for (std::uint32_t z = 1; z <= max_z; ++z) {
    const std::size_t first = tables->ref_offset_[z];
    const int nref = tables->nref_[z];
    const std::vector<int> counts =
        gaussian_counts(tables->ref_scalars_.data() + first * kRefColumns + kRefCn, nref, kRefColumns);
    for (int r = 0; r < nref; ++r) {
      if (counts[static_cast<std::size_t>(r)] != tables->ngw_[first + static_cast<std::size_t>(r)]) {
        corrupt("Gaussian counts of Z=" + std::to_string(z) + " do not match the reference CNs");
      }
    }
  }
  return tables;
}

std::shared_ptr<const D4ReferenceData> D4ReferenceData::embedded() {
  static const std::shared_ptr<const D4ReferenceData> tables = from_bytes(
      detail::d4_reference_blob, detail::d4_reference_blob_size, VANDERFAST_D4_REFERENCE_SHA256);
  return tables;
}

void D4ReferenceData::check(int z) const {
  if (!supports(z)) {
    fail(ErrorCode::UnsupportedElement, "atomic number " + std::to_string(z) +
                                            " is not supported by D4 (1.." + std::to_string(max_z_) + ")");
  }
}

int D4ReferenceData::nref(int z) const {
  check(z);
  return nref_[static_cast<std::size_t>(z)];
}

std::size_t D4ReferenceData::first_ref(int z) const {
  check(z);
  return ref_offset_[static_cast<std::size_t>(z)];
}

double D4ReferenceData::element(int z, ElementColumn column) const {
  check(z);
  return element_[static_cast<std::size_t>(z) * kElementColumns + static_cast<std::size_t>(column)];
}

}  // namespace vanderfast
