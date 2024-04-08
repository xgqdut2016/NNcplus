#include "tensor.h"
#include <stdexcept> // for std::out_of_range
#include <numeric>   // for std::accumulate

// Tensor类的构造函数实现
Tensor::Tensor(const std::vector<size_t> &shape)
    : shape_(shape)
{
    // 初始化列表 : shape_(shape) 是用于初始化类的成员变量的一种方式。
    // 具体来说，shape_ 是 Tensor 类的一个成员变量（很可能是 std::vector<size_t> 类型），而 shape 是传递给构造函数的参数。
    //  shape_(shape) 表示将 shape_ 成员变量初始化为 shape 参数的值。
    // 这样，在构造函数体执行之前，shape_ 就已经被赋予了传入的 shape 参数的值
    // 计算tensor的总大小，并分配内存
    size_t total_size = 1;
    for (size_t dim : shape_)
    { // 遍历容器（数组，向量）内元素
        total_size *= dim;
    }
    data_.resize(total_size); // data_是tensor.h里面Tensor类的私有变量，resize函数会改变向量大小
    zerosInit();
}

// 获取tensor的形状，第一个const表示shape()函数不能修改shape_，第二个const表示shape()不修改Tensor中的任何成员变量
const std::vector<size_t> &Tensor::shape() const
{
    return shape_;
}

// 获取tensor的大小（元素总数）
size_t Tensor::size() const
{
    return data_.size();
}

float *Tensor::pointer()
{
    return data_.data(); // 返回data_向量第一个元素的指针
}

// 辅助函数：将多维索引转换为一维索引
size_t Tensor::flatten_index(const std::vector<size_t> &indices) const
{
    size_t flat_index = 0;
    size_t multiplier = 1;
    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        flat_index += indices[i] * multiplier;
        multiplier *= shape_[i];
    }
    return flat_index;
}

// 访问tensor中的元素（包含边界检查）
float &Tensor::at(const std::vector<size_t> &indices)
{
    // 检查索引是否越界
    if (indices.size() != shape_.size())
    {
        throw std::out_of_range("Index dimensions do not match tensor shape.");
    }
    for (size_t i = 0; i < shape_.size(); ++i)
    {
        if (indices[i] >= shape_[i])
        {
            throw std::out_of_range("Index out of range.");
        }
    }
    // 转换索引并返回引用
    return data_[flatten_index(indices)];
}

const float &Tensor::at(const std::vector<size_t> &indices) const
{ // 实现细节晦涩难懂
    // 常量版本的at函数与上面的非常量版本类似，只是返回的是常量引用
    return const_cast<Tensor *>(this)->at(indices);
}

// 拷贝构造函数定义
Tensor::Tensor(const Tensor &other)
    : shape_(other.shape_), data_(other.data_)
{
    // 这里不需要额外代码，因为成员已经被初始化列表复制
}
// 移动构造函数定义
Tensor::Tensor(Tensor &&other) noexcept
    : shape_(std::move(other.shape_)), data_(std::move(other.data_))
{
    // 由于成员已经被初始化列表移动，这里不需要额外代码
}

// 拷贝赋值运算符定义
Tensor &Tensor::operator=(const Tensor &other)
{                       // this指向调用该函数的对象，比如书tensorA=tensorB,this指向tensorA
    if (this != &other) // 避免出现tensorA=tensorA这种情况
    {
        shape_ = other.shape_;
        data_ = other.data_;
    }
    return *this;
}
// 移动赋值运算符定义
Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this != &other)
    {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
    }
    return *this;
}
void Tensor::zerosInit()
{
    std::fill(data_.begin(), data_.end(), 0.0f);
}
void Tensor::incrementalInit()
{
    for (int i = 0; i < size(); i++)
    {
        data_[i] = static_cast<float>(i);
    }
}
// 这是因为data_是一个简单的vector，它会自动处理内存。
// 如果Tensor类需要管理更复杂的资源，则需要实现这些特殊成员函数。
