#include "core/fnn.h" // 包含线性层的头文件
#include <stdexcept>

FNN::FNN(std::vector<size_t> layerList)
{
    if (layerList.size() < 2)
    {
        throw std::invalid_argument("the length of layerList < 2, error");
    }
    // 构造函数，根据输入形状、隐藏层大小和输出大小初始化网络层
    size_t input_size = layerList[0]; // 输入大小即为输入向量的维度

    // 添加隐藏层
    for (size_t i = 1; i < layerList.size() - 1; i++)
    {
        size_t output_size = layerList[i];
        std::vector<size_t> hiddenShape = {input_size, output_size};
        layers_.push_back(Linear(hiddenShape)); // 添加线性层，push_back往vector末尾添加元素
        input_size = output_size;
    }

    // 添加输出层
    size_t output_size = layerList.back(); //.backl()返回最后一个元素
    std::vector<size_t> outputShape = {input_size, output_size};
    layers_.push_back(Linear(outputShape));
}

Tensor FNN::forward(Tensor &input)
{
    // 正向传播函数，接受输入向量，返回网络输出
    Tensor output = input;
    for (auto &layer : layers_)
    {
        output = layer.linearTransform(output); // 线性变换
    }
    return output;
}
