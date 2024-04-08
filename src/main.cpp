#include "tensor.h"
#include "omp_matrix.h"
#include "linear.h"
#include <omp.h>
#include <iostream>

int main()
{
    size_t batchsize = 4; // 使用int会发出警告
    size_t M = 64;
    size_t N = 3;
    // 创建一个2x3的tensor
    Tensor tensorA({batchsize, M});

    // 填充tensor的值
    tensorA.incrementalInit();

    Tensor tensorC({batchsize, N});

    double st, ela;
    st = omp_get_wtime();
    std::vector<size_t> linear_size({M, N});

    Linear mlp(linear_size);

    tensorC = mlp.linearTransform(tensorA);

    ela = omp_get_wtime() - st;
    for (size_t i = 0; i < tensorC.shape()[0]; ++i)
    {
        for (size_t j = 0; j < tensorC.shape()[1]; ++j)
        {
            std::cout << tensorC.at({i, j}) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << ela << std::endl;
    return 0;
}
