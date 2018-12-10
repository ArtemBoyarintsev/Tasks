#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/max_prefix_sum_cl.h"


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
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel max_prefix_sum_routine(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum_routine");
    ocl::Kernel max_prefix_sum_prepare(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum_prepare");
    max_prefix_sum_prepare.compile();
    max_prefix_sum_routine.compile();

    for (int n = 1; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = -2147000000;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = -2147000000;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            // TODO: implement on OpenCL
            auto resizeAndWrite = [](gpu::gpu_mem_32i* array_gpu, int size)
            {
                int init = 0;
                for (int i =0; i < size; ++i)
                {
                    array_gpu[i].resizeN(1);
                    array_gpu[i].writeN(&init, 1);
                }
            };

            unsigned int workGroupSize = 256;
            gpu::gpu_mem_32i as_gpu, work_horse_mem_gpu[2];
            as_gpu.resizeN(n);
            work_horse_mem_gpu[0].resizeN(3*(n / workGroupSize + 1));
            work_horse_mem_gpu[1].resizeN(3*(n / (workGroupSize * workGroupSize) + 1));
            as_gpu.writeN(as.data(), n);
            timer t;


            int arrayWithAnswer;
            for (int iter = 0; iter < benchmarkingIters; ++iter)
            {
                unsigned int global_work_size = ((n + workGroupSize - 1) / workGroupSize) * workGroupSize;
                max_prefix_sum_prepare.exec(
                        gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, work_horse_mem_gpu[0], n);

                int first = 0;
                for (int currentSize = n / workGroupSize; currentSize > 1; currentSize /= workGroupSize)
                {
                    unsigned int global_work_size = ((currentSize + workGroupSize - 1) / workGroupSize) * workGroupSize;
                    max_prefix_sum_routine.exec(
                            gpu::WorkSize(workGroupSize, global_work_size),
                            work_horse_mem_gpu[first], work_horse_mem_gpu[1-first], currentSize);

                    first = 1 - first;
                }
                arrayWithAnswer = first;
                t.nextLap();
            }
            int garbage_answer[3];
            work_horse_mem_gpu[arrayWithAnswer].readN(garbage_answer, 3);
            int gpu_max_sum = garbage_answer[1];
            int gpu_max_index = garbage_answer[2];
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
            EXPECT_THE_SAME(reference_result, gpu_max_index+1, "GPU result should be consistent!");
            EXPECT_THE_SAME(reference_max_sum, gpu_max_sum, "GPU result of max_sum should be consistent!");

        }
    }
}
