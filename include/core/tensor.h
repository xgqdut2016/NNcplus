#pragma once
#include <vector>
#include <cstddef> // for size_t

// 最简单的Tensor类声明
class Tensor
{
public:
    // 构造函数：接收一个形状向量来初始化tensor,&可以避免不必要的拷贝，提高效率
    Tensor(const std::vector<size_t> &shape);

    // 获取tensor的形状
    const std::vector<size_t> &shape() const; // shape()函数的返回对象的数据类型是const std::vector<size_t> &

    // 获取tensor中指定位置的元素值（假设已经进行了边界检查）
    float &at(const std::vector<size_t> &indices);             // 返回对象数据类型是float &，该函数允许修改tensor元素数值
    const float &at(const std::vector<size_t> &indices) const; // 只能访问，不能修改tensor元素

    // 获取tensor中元素的数量（即所有维度大小的乘积）
    size_t size() const;

    float *pointer(); // 返回的数据类型是float*，也就是数组的首地址指针
    // 拷贝构造函数声明
    Tensor(const Tensor &other); // 将另一个Tensor对象的内容复制一份来构造新对象

    // 移动构造函数声明，两个&是右值引用，将资源所有权移动到另一个对象，将另一个对象的资源（指针）移动到新的对象，性能更高
    Tensor(Tensor &&other) noexcept; // noexcept指示函数是否可能抛出异常，告诉编译器不要考虑异常，防止编译器自作聪明

    // 拷贝赋值运算符声明 tensorA = tensorB即可，即把tensorB的内容额外复制一份给tensorA
    Tensor &operator=(const Tensor &other);

    // 移动赋值运算符声明，不会额外复制tensorB，tensorA只是在tensorB的基础上移动了tensorB的指针得到
    Tensor &operator=(Tensor &&other) noexcept; // 使用方法是tensorA=std::move(tensorB)
    void zerosInit();
    void incrementalInit(); // for(i=0;i<n;i++)初始化数据为i

private:
    std::vector<size_t> shape_; // 存储tensor的形状
    std::vector<float> data_;   // 存储tensor的数据，这里简化为float类型

    // 辅助函数，用于将多维索引转换为平坦化的一维索引
    size_t flatten_index(const std::vector<size_t> &indices) const;
};
