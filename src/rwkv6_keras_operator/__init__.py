import os
assert 'KERAS_BACKEND' in os.environ
from keras import ops
from distutils.util import strtobool
ops_kernel = strtobool(os.environ.get('OPS_KERNEL','0'))

if not ops_kernel:
    if os.environ['KERAS_BACKEND'] == 'jax':
        from .jax_rwkv_kernel import RWKVKernelOperator as CudaOperator
    elif os.environ['KERAS_BACKEND'] == 'torch':
        from .torch_rwkv_kernel import RWKVKernelOperator as CudaOperator
    else:
        CudaOperator = None
else:
    CudaOperator = None

from .ops_rwkv_kernel import RWKVKernelOperator as OpsOperator


#print('cuda kernel:',CudaOperator)

class RWKVKernelOperator:
    def __init__(self,head_size,max_sequence_length,ops_loop=False):

        self.enbale_cuda =  CudaOperator is not None

        if self.enbale_cuda:
            self.cuda_operator = CudaOperator(head_size,max_sequence_length)
        
        self.ops_operator = OpsOperator(head_size,max_sequence_length)
        
        self.ops_loop = ops_loop

    def __call__(self, r, k, v, w, u, with_state = False, init_state = None, state_map = None):
        seq_len = r.shape[1]
        def call_parallel():
            if self.enbale_cuda:
                return self.cuda_operator(r=r,k=k,v=v,w=w,u=u,with_state=with_state,init_state=init_state,state_map=state_map)
            else:
                return self.ops_operator(r=r,k=k,v=v,w=w,u=u,with_state=with_state,init_state=init_state,state_map=state_map)
        def call_one_step():
            return self.ops_operator(r=r,k=k,v=v,w=w,u=u,with_state=with_state,init_state=init_state,state_map=state_map)
        if not self.ops_loop:
            return ops.cond(seq_len !=1 and not ops_kernel, call_parallel, call_one_step)
        else:
            return call_parallel()


#from .ops_rwkv_kernal import RWKVKernelOperator as OPSKernelOperator


"""
新增三个参数
return_state 布尔类型 是否返回最终的state,如果想自定义init_state也需要启用这个开关

init_state
    当init_state省缺时，则使用全零初始化BatchSize维度上的状态。
    形状: (state_kinds,num_heads,head_size, head_size)， 其中state_kinds为小于等于Batch_Size的正整数
    精度: 在r为fp16时 init_state为fp32 其余时候类型与r相同


state_map
    形状: (Batch_Size,)
    精度: int64, list[int]
    这个数组定义了state到r上每个Batch维度切片间的映射关系
    取值范围: [0, state_kinds)

返回:
    output, output_state 

def __call__(self,r, k, v, w, u, return_state=False, init_state=None, state_map=None):




"""