---
sidebar_position: 2
---

# usparse Auto-tuner
usparse 库自动性能调优工具。

## demo
### 编译 demo
```bash
cd tools/usparse_tuner
cmake -B ./build -S ./ -DGMRES_TOOLS_ENABLE=ON
cmake --build ./build 
```

编译完成后，可以在 tools/usparse_tuner/ 目录下找到 usparse_tuner_example_sqrt 可执行文件。
执行它，即可体验 demo。

### 输出

执行上面的 demo，它的一次运行的日志可能入下：
```log
$ ./build/tools/usparse_tuner/usparse_tuner_example_sqrt 

uSparse Tuner summary:
------------------------------------------------------------------------------------------
TuneCase     TuneParams                          Time(μs)   Iterations      Counters
------------------------------------------------------------------------------------------
TuneSqrt/1/3 <P1=8:P2=1:P3=6:P4=4:P5=9>              10988           10 Bandwidth: 1.599757Mi/s
TuneSqrt/1/3 <P1=8:P2=1:P3=6:P4=4:P5=10>              7340           10 Bandwidth: 2.394840Mi/s
TuneSqrt/1/3 <P1=8:P2=1:P3=6:P4=5:P5=9>               7434           10 Bandwidth: 2.364558Mi/s
TuneSqrt/1/3 <P1=8:P2=1:P3=6:P4=5:P5=10>              7364           10 Bandwidth: 2.387035Mi/s
TuneSqrt/1/3 <P1=8:P2=1:P3=12:P4=4:P5=9>              6922           10 Bandwidth: 3.385943Mi/s
TuneSqrt/1/3 <P1=8:P2=1:P3=12:P4=4:P5=10>             6453           10 Bandwidth: 3.632032Mi/s
TuneSqrt/1/3 <P1=8:P2=1:P3=12:P4=5:P5=9>              6537           10 Bandwidth: 3.585360Mi/s
...
TuneSqrt/2/6 <P1=16:P2=0:P3=6:P4=5:P5=9>             13865           10 Bandwidth: 2.113009Mi/s
TuneSqrt/2/6 <P1=16:P2=0:P3=6:P4=5:P5=10>            13809           10 Bandwidth: 2.121578Mi/s
TuneSqrt/2/6 <P1=16:P2=0:P3=12:P4=4:P5=9>            21583           10 Bandwidth: 1.628886Mi/s
TuneSqrt/2/6 <P1=16:P2=0:P3=12:P4=4:P5=10>           27660           10 Bandwidth: 1.271014Mi/s
TuneSqrt/2/6 <P1=16:P2=0:P3=12:P4=5:P5=9>            17566           10 Bandwidth: 2.001381Mi/s
TuneSqrt/2/6 <P1=16:P2=0:P3=12:P4=5:P5=10>           17539           10 Bandwidth: 2.004461Mi/s
------------------------------------------------------------------------------------------
```

## 原理
在我们做性能调试的时候，可能会针对多组测试算例进行测试，表格的第一列 TuneCase 就是测试 case 对应的多组算例的参数。 这里，我们测试了两组算例，对应的参数分别是（（1，3）、（2，6））。  
第二列 TuneParams 是我们需要调的参数及其对应的取值。 例如，这里需要调 P1、P2、...、P5这五个参数，由于每个参数都有自己的取值范围，他们一般是有限个数的。例如P1 可以取 8 和 16，P2 可以取 0 和 1，...等等。  
这里，我们可以穷举了这些参数的全部组合（即笛卡尔积），测试每一组算例下对应参数组合的性能情况，即可得到上表。
表的每一列对应一个算例和一组参数下的被测单元的运行时间（Time）、迭代次数（Iterations）和其他用户自定义参数（Counters）。
需要说明的是，这里的 Time 和用户自定义的性能参数，都是平均每次迭代的时间或者平均每次迭代的性能。

## 使用方法

可参考 `tools/usparse_tuner/example_main.cc` 文件。 如果你用过 google benchmark， 应该会对这里的 API 比较熟悉，因为我们在实现这个参数调优工具的时候，可以和 google benchmark 保持接口的类似，便于大家熟悉相关的 API。

### 定义被测试的函数

这个函数一般是一个函数模板，可以有多个模板参数。其中的模板参数是我们调优工具进行调优的对象。
我们期望通过确定这些模板参数，实现最优的性能（实际上是穷举所有的可能的参数，观察耗时多少，确定参数如何选取）。
目前，该工具支持迭代所有的参数组合，并输出不同参数组合下的被测函数的执行时间和性能信息。

### 定义调参主体：`USPARSE_TUNER_BENCH()`

这个宏，定义调参的主体，需要给 `USPARSE_TUNER_BENCH` 传递一个调参的名字。

宏里面需要定义一个模板函数（函数签名必须是 `static void utuner(usparse::tuner::States & states, ...)`）的形式，其中：
- 必须是 `static`，必须返回 `void`；
- 函数名称必须是 `utuner`;
- 函数的第一个参数必须是 `usparse::tuner::States &`;
- 第二个参数是 `tuple`，对应被测试函数的函数调用参数。我们称之为 argument list。

在该 utuner 函数中，需要对 states 进行遍历（这和 google benchmark 的用法是一样的），进行多次迭代，实现对被测试函数的多次调用。
在该 utuner 函数中，可选地，还可以自定义其他性能指标（如带宽等），我们称之为 Counter。 这个用法和 google benchmark 是一致的。 Counter 目前支持速率指标（可选1000进制和 1024 进制），在输出的 summary 报告中可以看到这些自定义的 Counter。

### 调参范围与配置 `USPARSE_TUNER_DEFINE_TUNE_RANGE()`

这个宏，定义调参的参数范围和配置，这里也需要给 `USPARSE_TUNER_DEFINE_TUNE_RANGE` 传递一个调参的名字，且需要和上一步的保持一致。 这个是 usparse-tuner 独有的。

然后，我们用 `usparse::tuner::ParamRange` 定义多组参数和各个参数的可能的取值。
- `usparse::tuner::ParamRange` 的第一个模板参数是被调参数的类型，随后的模板参数是该被调参数所有的可能的取值。
- 我们可以定义多个 `usparse::tuner::ParamRange`。一般是和上面的 `utuner` 函数的模板参数对应（包括类型和可能的取值）。

随后我们定义 config，即：`usparse::tuner::Config`。
它是一个模板类，可以接受多个类型（即 argument list）作为模板参数。 这里，这个类型即是前面提到的 Args 对应的类型（即 utuner 函数的接收的系列参数的类型）。
所以，config 的 arguments list类型 需要和 tuner 函数的参数类型保持一致。

另外，config 可以为空，我们将会构造一个默认的 config。 除此之外，config 中还可以:
- 设置迭代次数: `setIterations(int)`;
- 设置多个算例: 链式调用 `setArgs`，每次调用对应一组算例参数。

最后，调用 `USPARSE_TUNER_LAUNCH` 宏，这些需要被调的参数范围、config 整合到一起，这个是实际启动调参过程，其中：
- 第1个参数是调参名称，和前面的保持一致；
- 第2个参数是 Config 的类型；
- 第3个参数是 config 对象；
- 第4个参数是调参的组数；
- 后面的模板参数是各组参数的取值范围。

### `USPARSE_TUNER_MAIN`
这是 main 函数，执行前面定义的所有的调参，并打印调参结果报告。