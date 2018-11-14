#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

int find_begin_on_diag(const __global float* first,
                       const __global float* second,
                       int startDiagIndex, int count)
{
    if (startDiagIndex == 0)
        return 0;

    int secondIndex = max(0, startDiagIndex - count);
    int firstIndex = startDiagIndex - secondIndex;

    while (firstIndex - secondIndex > 0)
    {
        int middleIndex = secondIndex + (firstIndex - secondIndex) / 2;
        if (first[middleIndex] > second[startDiagIndex - 1 - middleIndex])
        {
            firstIndex = middleIndex;
        }
        else
        {
            secondIndex = middleIndex + 1;
        }
    }

    return secondIndex;
}


__kernel void simple_sort(__global float *a)
{
    __local float aLocalCopy[128]; //work_group_size is 128
    const unsigned local_id = get_local_id(0);
    const unsigned global_id = get_global_id(0);
    aLocalCopy[local_id] = a[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0)
    {
        for (int i = 0; i < 128; ++i)
        {
            for (int j = i + 1; j < 128; ++j)
            {
                if (aLocalCopy[i] > aLocalCopy[j])
                {
                    float k = aLocalCopy[i];
                    aLocalCopy[i] = aLocalCopy[j];
                    aLocalCopy[j] = k;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    a[global_id] = aLocalCopy[local_id];
}


__kernel void merge(__global const float* a,
                    __global float* res,
                     unsigned int n,
                     unsigned int countPerWorkGroup) // countPerWorkGroup in each subarray
{
    const unsigned group_id = get_group_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned work_group_size = get_local_size(0);

    const __global float* first = a + group_id * countPerWorkGroup * 2; // by 2, as we merge two arrays
    const __global float* second = first + countPerWorkGroup;

    int countPerWI = 2 * countPerWorkGroup / work_group_size; // count per work item in the result array

    int startIndex = (local_id) * countPerWI;

    int firstIndex = find_begin_on_diag(first, second, startIndex, countPerWorkGroup);
    int lastFirstIndex = firstIndex + countPerWI;
    if (lastFirstIndex > countPerWorkGroup)
        lastFirstIndex = countPerWorkGroup;

    int secondIndex = startIndex - firstIndex;
    int lastSecondIndex = secondIndex + countPerWI;

    if (lastSecondIndex > countPerWorkGroup)
        lastSecondIndex = countPerWorkGroup;

    int resIndex = startIndex;

    int offset = group_id * countPerWorkGroup * 2;
    for (int i = 0; i < countPerWI; ++i)
    {
        if (secondIndex == lastSecondIndex || firstIndex < lastFirstIndex && first[firstIndex] < second[secondIndex])
            res[offset + resIndex++] = first[firstIndex++];
        else
            res[offset + resIndex++] = second[secondIndex++];

        if (offset+resIndex == 509)
        {
            /*printf("%f\n", res[508]);
            printf("%f %f\n", first[firstIndex], first[firstIndex - 1]);
            printf("%f %f\n", second[secondIndex], second[secondIndex - 1]);
            printf("%d %d\n", firstIndex, secondIndex);*/
        }
    }
}
