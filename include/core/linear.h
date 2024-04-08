#pragma once
#include <vector>
#include "core/tensor.h"
class Linear
{
public:
    Linear(std::vector<size_t> shape); // 构造函数，接受形式如(M,N)的shape作为参数

    std::vector<size_t> shape_; // 权重的shape

    Tensor linearTransform(Tensor &Input); // 定义正向传播函数，接受一个类型为Tensor的类作为输入

    std::vector<float> weight_; // 权重的具体元素，形状为shape=[M,N]
    std::vector<float> bias_;   // 形状为[N]
};
