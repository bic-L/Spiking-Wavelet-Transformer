from abc import abstractmethod
from typing import Callable, overload
import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, base, lava_exchange
from spikingjelly import configure
import math
import numpy as np
import logging
try:
    import cupy
    from spikingjelly.clock_driven import neuron_kernel, cu_kernel_opt
except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cu_kernel_opt = None

try:
    import lava.lib.dl.slayer as slayer

except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    slayer = None


def check_backend(backend: str):
    if backend == 'torch':
        return
    elif backend == 'cupy':
        assert cupy is not None, 'CuPy is not installed! You can install it from "https://github.com/cupy/cupy".'
    elif backend == 'lava':
        assert slayer is not None, 'Lava-DL is not installed! You can install it from "https://github.com/lava-nc/lava-dl".'
    else:
        raise NotImplementedError(backend)


class NegBaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., neg_v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        """
        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``, 正或者负, 超过都fire
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        可微分SNN神经元的基类神经元。

        * :ref:`中文API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        This class is the base class of differentiable spiking neurons.
        """
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(neg_v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.register_memory('v_threshold', v_threshold)
        self.register_memory('neg_v_threshold', neg_v_threshold)
        self.register_memory('v_reset', v_reset)

        self.timestep = torch.tensor(0.).cuda()
        self.firing_rate = torch.tensor(0.).cuda()

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.register_memory('inhibit', 1e-3)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
         * :ref:`API in English <BaseNode.neuronal_charge-en>`

        .. _BaseNode.neuronal_charge-cn:

        定义神经元的充电差分方程。子类必须实现这个函数。

        * :ref:`中文API <BaseNode.neuronal_charge-cn>`

        .. _BaseNode.neuronal_charge-en:


        Define the charge difference equation. The sub-class must implement this function.
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """
        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-cn:

        根据当前神经元的电压、阈值，计算输出脉冲。

        * :ref:`中文API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-en:


        Calculate out spikes of neurons by their current membrane potential and threshold voltage.
        """

        pos_spike = self.surrogate_function(self.v - self.v_threshold)
        neg_spike = self.surrogate_function(-(self.v + self.neg_v_threshold  + self.inhibit))

        
        return pos_spike - neg_spike


    def neuronal_reset(self, spike):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:

        66249369-828f-46b6-af7c-8eac953b6ba9
        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        
        pos_spike = spike_d > 0
        neg_spike = spike_d < 0

        spike_d_vth = pos_spike * self.v_threshold + neg_spike * self.neg_v_threshold

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d_vth

        else:
            # hard reset
            self.v = (1. - spike_d.abs()) * self.v + spike_d * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, neg_v_threshold={self.neg_v_thresholdv_threshold},v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):
        """

        * :ref:`API in English <BaseNode.forward-en>`

        .. _BaseNode.forward-cn:

        :param x: 输入到神经元的电压增量
        :type x: torch.Tensor

        :return: 神经元的输出脉冲
        :rtype: torch.Tensor

        按照充电、放电、重置的顺序进行前向传播。

        * :ref:`中文API <BaseNode.forward-cn>`

        .. _BaseNode.forward-en:

        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes of neurons
        :rtype: torch.Tensor

        Forward by the order of `neuronal_charge`, `neuronal_fire`, and `neuronal_reset`.

        """
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        return spike


class NegIFNode(NegBaseNode):
    def __init__(self, v_threshold: float = 1., neg_v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, cupy_fp32_inference=False):
        """
        * :ref:`API in English <IFNode.__init__-en>`

        .. _IFNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param cupy_fp32_inference: 若为 `True`，在 `eval` 模式下，使用float32，却在GPU上运行，并且 `cupy` 已经安装，则会自动使用 `cupy` 进行加速
        :type cupy_fp32_inference: bool

        Integrate-and-Fire 神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + X[t]

        * :ref:`中文API <IFNode.__init__-cn>`

        .. _IFNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param cupy_fp32_inference: If `True`, if this module is in `eval` mode, using float32, running on GPU, and `cupy` is installed, then this
            module will use `cupy` to accelerate
        :type cupy_fp32_inference: bool

        The Integrate-and-Fire neuron, which can be seen as a ideal integrator. The voltage of the IF neuron will not decay
        as that of the LIF neuron. The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + X[t]

        """
        super().__init__(v_threshold, neg_v_threshold, v_reset, surrogate_function, detach_reset)

        if cupy_fp32_inference:
            check_backend('cupy')
        self.cupy_fp32_inference = cupy_fp32_inference
        self.momentum = 0.1


    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def forward(self, x: torch.Tensor):
        if self.cupy_fp32_inference and cupy is not None and not self.training and x.dtype == torch.float32:
            # cupy is installed && eval mode && fp32
            device_id = x.get_device()
            if device_id < 0:
                return super().forward(x)

            # use cupy to accelerate
            if isinstance(self.v, float):
                v = torch.zeros_like(x)
                if self.v != 0.:
                    torch.fill_(v, self.v)
                self.v = v

            if self.v_reset is None:
                hard_reset = False
            else:
                hard_reset = True

            code = rf'''
                extern "C" __global__
                void IFNode_{'hard' if hard_reset else 'soft'}_reset_inference_forward(
                const float * x, const float & v_threshold, const float & neg_v_threshold, {'const float & v_reset,' if hard_reset else ''}
                float * spike, float * v,
                const int & numel)
            '''

            code += r'''
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < numel)
                    {
                        v[index] += x[index];

                        float pos_spike = v[index] >= v_threshold;
                        float neg_spike = v[index] <= -( neg_v_threshold  + 1E-3);

                        spike[index] = pos_spike - neg_spike;

                        float spike_d_vth = pos_spike * v_threshold + neg_spike * neg_v_threshold;

            '''

            code += rf'''
                        {' v[index] = (1.0f - abs(spike_d[index])) * v[index] + spike_d[index] * v_reset;;' if hard_reset else 'v[index] -= spike_d_vth;'}
            '''

            code += r'''
                    }
                }
            '''
            if hasattr(self, 'cp_kernel'):
                if self.cp_kernel.code != code:
                    # replace codes
                    del self.cp_kernel
                    self.cp_kernel = cupy.RawKernel(code, f"IFNode_{'hard' if hard_reset else 'soft'}_reset_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)
            else:
                self.cp_kernel = cupy.RawKernel(code, f"IFNode_{'hard' if hard_reset else 'soft'}_reset_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)

            with cu_kernel_opt.DeviceEnvironment(device_id):
                numel = x.numel()
                threads = configure.cuda_threads
                blocks = cu_kernel_opt.cal_blocks(numel)
                cp_numel = cupy.asarray(numel)
                cp_v_threshold = cupy.asarray(self.v_threshold, dtype=np.float32)
                cp_v_threshold = cupy.asarray(self.v_threshold, dtype=np.float32)
                if hard_reset:
                    cp_v_reset = cupy.asarray(self.v_reset, dtype=np.float32)

                spike = torch.zeros_like(x)
                if hard_reset:
                    x, cp_v_threshold, cp_v_reset, spike, self.v, cp_numel = cu_kernel_opt.get_contiguous(x, cp_v_threshold, cp_v_reset, spike, self.v, cp_numel)
                    kernel_args = [x, cp_v_threshold, cp_v_reset, spike, self.v, cp_numel]
                else:
                    x, cp_v_threshold, spike, self.v, cp_numel = cu_kernel_opt.get_contiguous(x, cp_v_threshold, spike, self.v, cp_numel)
                    kernel_args = [x, cp_v_threshold, spike, self.v, cp_numel]
                self.cp_kernel(
                    (blocks,), (threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device_id,
                        *kernel_args
                    )
                )
                return spike
        else:
            return super().forward(x)


class MultiStepNegIFNode(NegIFNode):
    def __init__(self, v_threshold: float = 0.5, neg_v_threshold: float = 0.5, v_reset: float = 0., momentum: float = 0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6):
        """
        * :ref:`API in English <MultiStepIFNode.__init__-en>`

        .. _MultiStepIFNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param backend: 使用哪种计算后端，可以为 ``'torch'`` 或 ``'cupy'``。``'cupy'`` 速度更快，但仅支持GPU。
        :type backend: str

        多步版本的 :class:`spikingjelly.clock_driven.neuron.IFNode`。

        .. tip::

            对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
            使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepIFNode.__init__-cn>`

        .. _MultiStepIFNode.__init__-en:

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        The multi-step version of :class:`spikingjelly.clock_driven.neuron.IFNode`.

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.

        """
        super().__init__(v_threshold, neg_v_threshold, v_reset, surrogate_function, detach_reset)

        self.momentum = momentum
        
        self.register_memory('v_seq', None)

        check_backend(backend)

        self.backend = backend

        self.lava_s_cale = lava_s_cale

        if backend == 'lava':
            self.lava_neuron = self.to_lava()
        else:
            self.lava_neuron = None


    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))

            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            
            
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepNegIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.v_threshold, self.neg_v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)
  
            return spike

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

    def to_lava(self):
        return lava_exchange.to_lava_neuron(self)


    def reset(self):
        super().reset()
        if self.lava_neuron is not None:
            self.lava_neuron.current_state.zero_()
            self.lava_neuron.voltage_state.zero_()
