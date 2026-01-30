// Minimal compilation check for codegen ABIs.

#include "pto/wsp/codegen/abi/kernel_abi.hpp"
#include "pto/wsp/codegen/abi/workload_abi.hpp"
#include "pto/wsp/codegen/abi/ptoisa_bridge.hpp"

static_assert(sizeof(KernelTaskDesc) > 0);
static_assert(sizeof(RuntimeContext) > 0);
static_assert(sizeof(CSPTContext) > 0);

int main() { return 0; }

