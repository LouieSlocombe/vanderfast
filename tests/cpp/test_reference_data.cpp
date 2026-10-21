#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include "vanderfast/build_info.hpp"
#include "vanderfast/error.hpp"
#include "vanderfast/reference_data.hpp"

using vanderfast::Error;
using vanderfast::ReferenceData;

namespace {

std::vector<unsigned char> blob() {
  std::ifstream file(VANDERFAST_REFERENCE_BLOB_PATH, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST_CASE("embedded tables match upstream spot values") {
  auto tables = ReferenceData::embedded();
  CHECK(tables->max_z() == 103);
  CHECK(tables->max_ref() == 7);
  CHECK(tables->identity() == VANDERFAST_REFERENCE_SHA256);
  CHECK(tables->source_id().rfind("simple-dftd3 v1.6.0", 0) == 0);
  CHECK(tables->nref(1) == 2);
  CHECK(tables->nref(96) == 6);
  CHECK(tables->reference_cn(1, 0) == 0.9118);
  CHECK(tables->reference_cn(101, 5) == -1.0);
  CHECK(tables->c6(1, 1, 0, 0) == 3.0267);
  CHECK(tables->c6(1, 1, 1, 0) == 4.7379);
  CHECK(tables->c6(1, 1, 1, 1) == 7.5916);
  CHECK(tables->c6(6, 1, 4, 1) == tables->c6(1, 6, 1, 4));
  CHECK(tables->rvdw(1, 8) == tables->rvdw(8, 1));
  CHECK(tables->rcov(1) == (4.0 / 3.0) * (1.8897261246204404 * 0.32));
}

TEST_CASE("accessors reject unsupported elements and references") {
  auto tables = ReferenceData::embedded();
  CHECK_FALSE(tables->supports(0));
  CHECK_FALSE(tables->supports(104));
  CHECK_THROWS_AS(tables->nref(104), Error);
  CHECK_THROWS_AS(tables->reference_cn(1, 2), Error);
  CHECK_THROWS_AS(tables->c6(1, 1, 0, 2), Error);
}

TEST_CASE("from_bytes validates the structure") {
  std::vector<unsigned char> data = blob();
  REQUIRE(data.size() > 512);
  auto copy = ReferenceData::from_bytes(data.data(), data.size(), "copy");
  CHECK(copy->identity() == "copy");
  CHECK(copy->c6(26, 8, 2, 1) == ReferenceData::embedded()->c6(26, 8, 2, 1));

  CHECK_THROWS_AS(ReferenceData::from_bytes(data.data(), 100, ""), Error);
  auto corrupt = [&](std::size_t offset, unsigned char value) {
    std::vector<unsigned char> bad = data;
    bad[offset] = value;
    return ReferenceData::from_bytes(bad.data(), bad.size(), "");
  };
  CHECK_THROWS_AS(corrupt(0, 'X'), Error);    // magic
  CHECK_THROWS_AS(corrupt(8, 2), Error);      // version
  CHECK_THROWS_AS(corrupt(20, 0), Error);     // flags
  CHECK_THROWS_AS(corrupt(36, 7), Error);     // pair count
  CHECK_THROWS_AS(corrupt(112, 5), Error);    // section count
  CHECK_THROWS_AS(corrupt(124, 9), Error);    // dtype
  CHECK_THROWS_AS(corrupt(128, 1), Error);    // misaligned offset
  std::uint64_t nref_offset = 0;
  std::memcpy(&nref_offset, data.data() + 128, sizeof(nref_offset));
  CHECK_THROWS_AS(corrupt(nref_offset + 1, 9), Error);  // reference count out of range
  CHECK_THROWS_AS(corrupt(nref_offset, 1), Error);      // slot 0 must be zero
}
