#pragma once
#include "core/linear.h" // 包含线性层的头文件

class FNN
{
public:
    FNN(std::vector<size_t> layerList); // layerList=[2,..., 1]表示输入节点为2，输出节点为1
    Tensor forward(Tensor &input);      // 那么对应的input的形状就是[batchsize, 2]

private:
    std::vector<Linear> layers_; // layers_是一个vector，但是这个vector的每个元素都是一个线性层，类似于{[2,.],[.,.],...,[.,1]}
};
