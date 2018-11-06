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
    int firstIndex = startDiagIndex - 1 - secondIndex;

    int minIndex = secondIndex;
    int maxIndex = firstIndex;

    while (firstIndex - secondIndex > 2)
    {
        int middleIndex = secondIndex + (firstIndex - secondIndex) / 2;
        if (first[middleIndex] >= second[startDiagIndex - 1 - middleIndex])
        {
            firstIndex = middleIndex + 1;
        }
        else
        {
            secondIndex = middleIndex;
        }
    }
    secondIndex = startDiagIndex - 1 - firstIndex;
    if (first[firstIndex] >= second[secondIndex])
    {
        while (firstIndex >=minIndex && secondIndex <= maxIndex && first[firstIndex] >= second[secondIndex])
        {
            firstIndex--;
            secondIndex++;
        }
        return firstIndex + 1;
    }
    else
    {
        while (firstIndex <= maxIndex && secondIndex >= minIndex && first[firstIndex] < second[secondIndex])
        {
            firstIndex++;
            secondIndex--;
        }
        return firstIndex;
    }
    //printf("%f %f %f %f\n", first[firstIndex], second[secondIndex], first[firstIndex+1], second[secondIndex-1]);
    return firstIndex + 1;
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
