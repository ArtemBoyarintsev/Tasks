#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"


#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 1024*1024*32;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;
   /* for (int i = 0; i < n / 128 + (n % 128 != 0); ++i)
    {
        int length = (i == (n / 128) + 1) ? n % 128 : 128;
        std::qsort(as.data() + 128 * i, length , sizeof(int), [](const void* a, const void* b)
        {
            float arg1 = *static_cast<const float*>(a);
            float arg2 = *static_cast<const float*>(b);
            if(arg1 < arg2) return -1;
            if(arg1 > arg2) return 1;
            return 0;
        });
    }*/
    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu, res_gpu;
    as_gpu.resizeN(n);
    res_gpu.resizeN(n);
    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        ocl::Kernel simple_sort(merge_kernel, merge_kernel_length, "simple_sort");
        merge.compile();
        simple_sort.compile();

        timer t;
        unsigned int workGroupSize = 128;

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            simple_sort.exec(gpu::WorkSize(workGroupSize, n), as_gpu);
            for (unsigned count = 128; count < n; count *= 2)
            {
                int groupCount = n / (2 * count);
                merge.exec(gpu::WorkSize(workGroupSize, groupCount * workGroupSize), as_gpu, res_gpu, n, count);
                res_gpu.readN(as.data(), n);

                as_gpu.swap(res_gpu);
            }
            as_gpu.swap(res_gpu);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        res_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}