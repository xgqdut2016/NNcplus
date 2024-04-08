# NNcplus
Building a Deep Learning Inference Framework with C++Code
## C++构建项目框架如下：
1：新建一个include文件夹，该文件夹主要存放各种.h，.hpp等头文件，头文件用于申明各种函数，各种类
2：新建一个src文件夹，该文件夹主要存放各种.cpp，.cc文件，主要负责实现头文件里面申明的函数和类
3：新建一个CMakeLists.txt文件，这个文件编写相关的编译原则
4：新建一个build文件夹，该文件夹一开始是空的，等待上面提到的include,src文件夹都填充好了相关的文件，CMakeLists.txt也编写完成
## 编译过程如下：
1：进入build文件夹
```bash
cd build
```
2：使用cmake命令生成Makefile文件以及其他文件
```bash
cmake ../
```
3：使用ls查看build文件夹确定Makefile文件存在以后，使用make命令自动寻找当前文件夹的Makefile文件编译
```bash
make
```
4：编译结束以后使用ls查看当前文件夹是否生成可执行文件（比如说可执行文件为tensor），然后运行可执行文件即可
```bash
./tensor
```
