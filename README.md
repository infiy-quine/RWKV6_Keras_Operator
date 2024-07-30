
## 基本介绍：
&nbsp;&nbsp;&nbsp;&nbsp;这是一个适用于bert4keras3库中RWKV6模块的rwkv核算子。这个算子在jax、pytorch框架提供了原生CUDA实现，但对于TensorFlow框架只提供基本的上层api实现。
## 安装
`在虚拟环境内执行 pip install rwkv6-keras-operator 安装算子，并阅读对应的注意事项。`
## pytorch使用注意事项:
- 安装依赖项 keras ninja 完整的cuda工具包
- 如果您使用vs code搭配虚拟环境进行调试，请确保终端在运行代码之前已经进入到了虚拟环境之中（而非依赖于vs code自动进入），以防ninja无法正常工作。
- 虽然Pytorch遇到"虚拟环境中的CUDA版本"与"全局CUDA环境版本"不一致时，rwkv6算子仍能正常工作，但是仍强烈建议将两环境的版本保持一致。
- 因为Pytorch的限制，无法在单一程序内实例化多个RWKVKernelOperator对象，因此请确保在同一个程序中只实例化一个rwkv6算子。但是算子是线程安全的（无状态的），可以放心的在不同位置调用。

## jax使用注意事项:
- 安装依赖项 keras gcc pybind11 完整的cuda工具包
- 如果您是通过虚拟环境的方式为jax安装cuda，您仍需要在本机中安装一个完整的CUDA环境（同Pytorch）。此外为了保证jax的并行编译(提升编译速度)能正常工作，您确保虚拟环境中的CUDA工具包与全局CUDA工具包的版本保持一致。
- jax编译依赖于/usr/local/cuda超链接，这个超链接在大部分情况下会自动创建，但是如果超链接不存在请手动指向程序的根目录。
  例如：
  ```shell
  example@example:~/STAR-RWKV6$ ls /usr/local/cuda -al
  lrwxrwxrwx 1 root root 21 Jun 16 09:37 /usr/local/cuda -> /usr/local/cuda-12.4/
  ```
- 此外请确保`nvcc -V`可以正确输出，并且`which nvcc`指向了正确的cuda版本。
- - 因为Jax的限制，无法在单一程序内实例化多个RWKVKernelOperator对象，请确保在同一个程序中只实例化一个rwkv6算子。但是算子是线程安全的（无状态的），可以放心的在不同位置调用。

## tensorflow使用注意事项：
- tensorflow只实现了基于原生API的RWKV6算子，这个算子只能用于模型的推理并且效率比较低。

## 使用方法：
### 配置环境变量
- 同keras3一样需要通过虚拟环境手动指定keras3的后端。  
  jax:
  ```python
  import os
  os.environ['KERAS_BACKEND'] = 'jax'
  ```
  pytorch:
  ```python
  import os
  os.environ['KERAS_BACKEND'] = 'pytorch'
  ```
  tensorflow:
  ```python
  import os
  os.environ['KERAS_BACKEND'] = 'tensorflow'
  ```
### 方法定义
- 使用`rwkv6_keras_operator import RWKVKernelOperator`导入算子，这个算子需要两个固定参数`head_size`和`max_sequence_length`,和一个可选参数`ops_loop`。
  - `head_size`为rwkv6的头大小，如果不清楚模型的头大小可以直接填64（在大部分情况下都是正确的）。
  - `max_sequence_length`为训练过程中的序列的最大长度，推理过程中的序列长度不受这个参数的限制。
  - 上面的参数均为必填项，并且会被以常量的形式编译到算子中。
  - 下面的`ops_loop`为可选项，这个参数的作用是当序列长度为1时（生成阶段）使用上层API的实现处理数据（代替CUDA算子）
  ```python
  operator = RWKVKernelOperator(head_size=64,max_sequence_length=4096, ops_loop=False)
  ```

- operator对象是callable的，通过operator(xxxx)调用算子。  
  def __call__(self, r, k, v, w, u, with_state=False, init_state=None, state_map=None) -> tensor, Union[tensor, None]:
  - `r`, `k`, `v`, `w` 的形状均为（batch_size, seq_len，hidden_size）  
  #batch_size=批次大小，seq_len=序列长度，hidden_size=隐藏层维度
  - `w`为重参数化前的输入，即`exp(-exp(w))`在已经由算子内部完成，不需要自己完成。
  - `u`的形状为(num_heads, head_size)或者形状为(hidden_size,)均可。
  - `with_state`代表“开启自定义初始状态”或“输出结束状态”。
  - `init_state`为初始状态，若这个值不为None，`with_state`的值必须为True，`init_state`的形状为（n_state, num_heads, head_size, head_size），
  - n_state为状态数。若这个值为1，则所有样本使用相同的自定义初始状态，若值n_state等于batch_size，则状态与输入数据为一一对应关系。

  - `state_map`为数据类型是int32，形状为（batch_size,）的一维数值，这个矩阵定义了`init_state`到数据的映射关系（存放了`init_state`在最初维度的切片索引），每个切片数据都会从`state_map`对应位置获取值（`init_state`的下标），进而从`init_state`中获取数据。

- 返回值由两个组成计算结果`y`，输出状态`y_state`，`with_state`值为False时，`y_state`为None。
  - `y`的形状为(batch_size, seq_len, hidden_size)
  - `y_state`的形状为(batch_size, num_heads, head_size, head_size) [或为None]
## 环境变量
- `OPS_KERNEL`默认为0，可以为0或1。如果这个环境变量的值为1强制使用基于上层API的算子代替基于底层实现的CUDA算子。这个环境变量必须在导入工具包之前设置，才能生效。
## 小贴士：
- 算子本身没有实现分布式支持，pytorch基于多线程的分布式可以直接适配使用，但是如果是基于sharded tensor的jax实现分布式则需要通过shard_map对算子进行包装（例子如下）。
  ```python
  import math
  import os
  os.environ['KERAS_BACKEND'] = 'jax'
  import jax
  from jax.experimental.shard_map import shard_map
  from jax.experimental.mesh_utils import create_device_mesh
  from jax.sharding import PartitionSpec as P
  from rwkv6_keras_operator import RWKVKernelOperator
  from jax.sharding import Mesh, NamedSharding
  from functools import partial
  import jax.numpy as jnp

  batch_size = 24
  head_size = 64
  num_heads = 32
  seq_length = 512
  hidden_size = head_size * num_heads
  mesh = Mesh(jax.devices('gpu'), axis_names=('device_axis',))
  num_devices = mesh.size
  
  device_p = P('device_axis')
  device_ns = NamedSharding(mesh, device_p)
  @partial(shard_map, mesh=mesh, in_specs=(device_p, device_p, device_p, device_p, device_p), out_specs=(device_p, device_p), check_rep=False)
  def call_kernel(r, k, v, w, u):
      print(r.shape)
      # 输入的形状为(1, batch_size, seq_len, hidden_size)， 需要降维度
      if len(r.shape) == 4: 
          r = jnp.squeeze(r, axis=0)
          k = jnp.squeeze(k, axis=0)
          v = jnp.squeeze(v, axis=0)
          w = jnp.squeeze(w, axis=0)
          u = jnp.squeeze(u, axis=0)
          y, ys =  operator(r, k, v, w, u, with_state=True)
          y = jnp.expand_dims(y, axis=0)
          ys = jnp.expand_dims(ys, axis=0)
          return y, ys
      else:
          return operator(r, k, v, w, u, with_state=True)
  
  operator = RWKVKernelOperator(head_size=head_size, max_sequence_length=seq_length)
  # max_sequence_length在前向过程中无用，但在反向传播过程中有用

  inputs_r = jax.random.normal(jax.random.PRNGKey(0), (num_devices, batch_size, seq_length, hidden_size))
  inputs_r = jax.device_put(inputs_r, device_ns)
  # 您可以把device作为一个单独的维度传入，也可以把他和batch_size合并成一个维度传入
  # inputs_r = jax.random.normal(jax.random.PRNGKey(0), (num_devices*batch_size, seq_length, hidden_size))
  # inputs_r = jax.device_put(inputs_r, device_ns)

  inputs_k = jax.random.normal(jax.random.PRNGKey(0), (num_devices, batch_size, seq_length, hidden_size))
  inputs_k = jax.device_put(inputs_k, device_ns)

  inputs_v = jax.random.normal(jax.random.PRNGKey(0), (num_devices, batch_size, seq_length, hidden_size))
  inputs_v = jax.device_put(inputs_v, device_ns)

  inputs_w = jax.random.normal(jax.random.PRNGKey(0), (num_devices, batch_size, seq_length, hidden_size))
  inputs_w = jax.device_put(inputs_w, device_ns)

  inputs_u = jax.random.normal(jax.random.PRNGKey(0), (num_devices, hidden_size))
  inputs_u = jax.device_put(inputs_u, device_ns)

  # 解冻下面的代码可以为上面的代码开启编译
  # call_kernel = jax.jit(call_kernel, in_shardings=(
  #     device_ns, device_ns, device_ns, device_ns, device_ns),
  #     out_shardings=(device_ns, device_ns))

  outputs_y, y_state = call_kernel(inputs_r, inputs_k, inputs_v, inputs_w, inputs_u)

  print(outputs_y.shape, outputs_y.sharding)
  print(y_state.shape, y_state.sharding)
  ```

- 控制台输出:
  ```shell
  (1, 24, 512, 2048)
  (8, 24, 512, 2048) NamedSharding(mesh=Mesh('device_axis': 8), spec=PartitionSpec('device_axis',))
  (8, 24, 32, 64, 64) NamedSharding(mesh=Mesh('device_axis': 8), spec=PartitionSpec('device_axis',))
  ```