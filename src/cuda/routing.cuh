/* Group token/choice slots and sum BF16 contributions in expert/choice order. */
__global__ void assign_groups_kernel(int slots_count, int expert_start, int groups, int max_rows,
                                     const int64_t *experts, const int32_t *active, int32_t *counts,
                                     int32_t *slots, int32_t *reverse)
{
    for (int64_t slot = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; slot < slots_count;
         slot += int64_t(gridDim.x) * blockDim.x) {
        int64_t expert = experts[slot];
        if (expert < expert_start || expert >= INT_MAX)
            continue;
        int local = int(expert) - expert_start;
        for (int group = 0; group < groups; ++group) {
            if (active[group] == local) {
                int row = atomicAdd(counts + group, 1);
                if (row < max_rows) {
                    slots[size_t(group) * max_rows + row] = int(slot);
                    reverse[slot] = group * max_rows + row;
                }
                break;
            }
        }
    }
}

__global__ void reduce_groups_kernel(int tokens, int choices, int width, int grouped_rows,
                                     const int64_t *experts, const int32_t *reverse,
                                     const __nv_bfloat16 *input, float *output)
{
    size_t count = size_t(tokens) * width;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += size_t(gridDim.x) * blockDim.x) {
        int token = int(index / width), column = int(index % width);
        int previous = -1, previous_choice = -1;
        float sum = 0;
        for (int step = 0; step < choices; ++step) {
            int smallest = INT_MAX, selected = -1;
            for (int top = 0; top < choices; ++top) {
                int64_t expert = experts[size_t(token) * choices + top];
                if (expert >= 0 && expert < INT_MAX &&
                    (expert > previous || (expert == previous && top > previous_choice)) &&
                    (expert < smallest || (expert == smallest && selected < 0))) {
                    smallest = int(expert);
                    selected = top;
                }
            }
            if (selected < 0)
                break;
            int row = reverse[size_t(token) * choices + selected];
            if (row >= 0 && row < grouped_rows)
                sum = __fadd_rn(sum, __bfloat162float(input[size_t(row) * width + column]));
            previous = smallest;
            previous_choice = selected;
        }
        output[index] = sum;
    }
}
