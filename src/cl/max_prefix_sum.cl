#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define LOCAL_MEMORY_MAX_SIZE 256*3


__kernel void max_prefix_sum_prepare(const __global int* global_a, __global int* global_out,
                    unsigned int size)
{
    __local int local_memory[LOCAL_MEMORY_MAX_SIZE];

    int values_per_group = get_local_size(0);

    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    int groupId = get_group_id(0);
    int numGroups = get_num_groups(0);

    if (globalId >= size)
    {
        local_memory[localId] = 0;
    }
    else
    {
        local_memory[localId] = global_a[groupId * values_per_group + localId];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0)
    {
        int sum = 0;
        int max_sum = -2147000000;
        int max_sum_index = 0;
        for (int i = 0; i < values_per_group; ++i)
        {
            sum += local_memory[i];
            if (sum > max_sum)
            {
                max_sum = sum;
                max_sum_index = groupId * values_per_group + i;
                //printf("%d %d\n", max_sum, max_sum_index);
            }
        }
        global_out[3*groupId] = sum;
        global_out[3*groupId + 1] = max_sum;
        global_out[3*groupId + 2] =  max_sum_index;
    }
}


__kernel void max_prefix_sum_routine(const __global int* global_a, __global int* global_out, unsigned int size)
{
    __local int local_memory[LOCAL_MEMORY_MAX_SIZE];

    int values_per_group = get_local_size(0);

    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    int groupId = get_group_id(0);
    int numGroups = get_num_groups(0);

    if (globalId >= size)
    {
        local_memory[3*localId] = 0;
        local_memory[3*localId + 1] = 0;
        local_memory[3*localId + 2] = 0;
    }
    else
    {
        local_memory[3 * localId] = global_a[3 * globalId];
        local_memory[3 * localId + 1] = global_a[3 * globalId + 1];
        local_memory[3 * localId + 2] = global_a[3 * globalId  + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0 && globalId < size)
    {
        int sum = local_memory[0];
        int max_sum = local_memory[1];
        int max_sum_index = local_memory[2];
        for (int i = 1; i < values_per_group; ++i)
        {
            int sum_and_max = sum + local_memory[3 * i + 1];
            if (sum_and_max > max_sum)
            {
                max_sum = sum_and_max;
                max_sum_index = local_memory[3 * i + 2];
            }
            sum += local_memory[3 * i];
        }

        global_out[3*groupId] = sum;
        global_out[3*groupId + 1] = max_sum;
        global_out[3*groupId + 2] = max_sum_index;
    }
}
