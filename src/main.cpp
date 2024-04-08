#include "core/tensor.h"
#include "cpu/omp_matrix.h"
#include "core/fnn.h"
#include <omp.h>
#include <iostream>

int main()
{
    size_t batchsize = 4; // 使用int会发出警告
    size_t M = 64;
    size_t N = 3;
    // 创建一个2x3的tensor
    Tensor tensorA({batchsize, M}); //{batchsize, M}是初始化列表(list)，可以初始化包括std::vector在内的各种类型的对象

    // 填充tensor的值
    tensorA.incrementalInit();

    Tensor tensorC({batchsize, N});

    double st, ela;
    st = omp_get_wtime();
    std::vector<size_t> layerList({M, 10, N});

    FNN mlp(layerList);

    tensorC = mlp.forward(tensorA);

    ela = omp_get_wtime() - st;
    for (size_t i = 0; i < tensorC.shape()[0]; ++i)
    {
        for (size_t j = 0; j < tensorC.shape()[1]; ++j)
        {
            std::cout << tensorC.at({i, j}) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "tensorC shape"
              << "[" << tensorC.shape()[0] << "," << tensorC.shape()[1] << "]"
              << " " << ela << std::endl;
    return 0;
}
