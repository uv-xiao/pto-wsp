// Minimal compilation check for codegen ABIs.

#include "pto/rt/codegen/abi/kernel_abi.hpp"
#include "pto/rt/codegen/abi/workload_abi.hpp"
#include "pto/rt/codegen/abi/ptoisa_bridge.hpp"

static_assert(sizeof(KernelTaskDesc) > 0);
static_assert(sizeof(RuntimeContext) > 0);
static_assert(sizeof(CSPTContext) > 0);

int main() { return 0; }

