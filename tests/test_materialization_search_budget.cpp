// Regression: the materialization search budget must scale with the incumbent it is trying to
// beat, not be pinned to a fixed few milliseconds.
//
// THE DEFECT. search_budget_ns was min(5 ms, incumbent_cost / 20). The proportional term is the
// intended governor -- spend up to 5% of what the fallback would cost looking for something
// better -- but a 5 ms ceiling overrides it for any incumbent above 100 ms, which is every real
// request. Measured on one RTX 5090 serving Qwen3.8-27B before the fix: stop_reason=time_budget
// on 439 of 444 requests, targets_evaluated falling from 45 to 14 as the conversation grew, and
// prefix reuse decaying from 98.5% on small turns to 13.6% at 188k tokens, because the search
// was cut off before it reached the reusable frontier. A completed search at that size takes
// about 9 ms, so a 5 ms ceiling terminates it just short of the answer.
//
// These assertions pin the POLICY, not the literal: below the ceiling the proportional term must
// govern (so cheap requests are unaffected), and for a realistically expensive incumbent the
// budget must comfortably exceed what a real search needs.

#include "runtime/engine/materialization_planner.h"

#include <cstdint>
#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

constexpr std::uint64_t kMillisecond = 1'000'000ULL;
constexpr std::uint64_t kSecond      = 1'000'000'000ULL;

// What the search actually needed at a 144k-token conversation, measured. The old 5 ms ceiling
// sat below this, which is precisely why the search terminated early.
constexpr std::uint64_t kObservedSearchCostNs = 9 * kMillisecond;

// The ceiling the defect shipped with. Kept here so the regression is stated, not implied.
constexpr std::uint64_t kOldCeilingNs = 5 * kMillisecond;

void test_proportional_term_governs_cheap_incumbents() {
    // A 20 ms incumbent is not worth 5 ms of searching; 5% of it is 1 ms and that must win.
    const std::uint64_t cheap = 20 * kMillisecond;
    expect(ninfer::runtime::materialization_search_budget_ns(cheap) == cheap / 20U,
           "cheap incumbent must be governed by the proportional term, not the ceiling");
    expect(ninfer::runtime::materialization_search_budget_ns(cheap) < kOldCeilingNs,
           "cheap incumbent must not have been widened by raising the ceiling");
    expect(ninfer::runtime::materialization_search_budget_ns(0) == 0,
           "a zero-cost incumbent warrants no search");
}

void test_expensive_incumbent_gets_a_usable_budget() {
    // A 144k-token root re-prefill. This is the live failing case.
    const std::uint64_t incumbent = 80 * kSecond;
    const std::uint64_t budget    = ninfer::runtime::materialization_search_budget_ns(incumbent);

    expect(budget > kOldCeilingNs,
           "REGRESSION: an 80 s incumbent must budget more than the old 5 ms ceiling");
    expect(budget > kObservedSearchCostNs,
           "REGRESSION: the budget must exceed what a real search costs (~9 ms), or the search "
           "terminates on TimeBudget before reaching the reusable frontier");
    expect(budget == ninfer::runtime::kMaterializationSearchCeilingNs,
           "an incumbent this expensive should be bounded by the ceiling");
}

void test_ceiling_is_a_guard_not_the_governor() {
    // The crossover: below it the proportional term wins, above it the ceiling clamps. The
    // defect was that this crossover sat at 100 ms, so the ceiling governed essentially always.
    const std::uint64_t crossover = ninfer::runtime::kMaterializationSearchCeilingNs * 20U;
    expect(ninfer::runtime::materialization_search_budget_ns(crossover - 20U) <
               ninfer::runtime::kMaterializationSearchCeilingNs,
           "just below the crossover the proportional term must still govern");
    expect(ninfer::runtime::materialization_search_budget_ns(crossover * 2U) ==
               ninfer::runtime::kMaterializationSearchCeilingNs,
           "well above the crossover the ceiling must clamp");
    expect(crossover > 1 * kSecond,
           "the crossover must sit above a second of incumbent cost, so that ordinary requests "
           "are governed proportionally rather than clamped");
}

void test_budget_is_monotonic() {
    std::uint64_t previous = 0;
    for (std::uint64_t incumbent = 0; incumbent <= 20 * kSecond; incumbent += kSecond) {
        const std::uint64_t budget = ninfer::runtime::materialization_search_budget_ns(incumbent);
        expect(budget >= previous, "budget must not decrease as the incumbent gets more expensive");
        previous = budget;
    }
}

} // namespace

int main() {
    test_proportional_term_governs_cheap_incumbents();
    test_expensive_incumbent_gets_a_usable_budget();
    test_ceiling_is_a_guard_not_the_governor();
    test_budget_is_monotonic();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
