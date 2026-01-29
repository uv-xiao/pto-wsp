// MARK_CPP_BODY_KERNEL_SNIPPET
float* x_ptr = (float*)task->tensor_ptrs[0];
float* y_ptr = (float*)task->tensor_ptrs[1];
const uint64_t xs3 = task->tensor_strides[0];
const uint64_t xs4 = task->tensor_strides[1];
const uint64_t ys3 = task->tensor_strides[2];
const uint64_t ys4 = task->tensor_strides[3];
for (int r = 0; r < 4; ++r) {
  for (int c = 0; c < 4; ++c) {
    y_ptr[r * (int64_t)ys3 + c * (int64_t)ys4] = x_ptr[r * (int64_t)xs3 + c * (int64_t)xs4];
  }
}

