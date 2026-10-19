"""Minimal TensorRT 10 engine runner using torch CUDA tensors as I/O buffers."""
import numpy as np
import tensorrt as trt
import torch

_TRT2TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
}


class TRTRunner:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inputs, self.outputs, self.buffers = [], [], {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _TRT2TORCH[self.engine.get_tensor_dtype(name)]
            buf = torch.empty(shape, dtype=dtype, device="cuda")
            self.buffers[name] = buf
            self.ctx.set_tensor_address(name, buf.data_ptr())
            mode = self.engine.get_tensor_mode(name)
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(name)

    def input_shape(self, idx=0):
        return tuple(self.buffers[self.inputs[idx]].shape)

    def __call__(self, *arrays):
        """arrays: numpy or torch tensors in engine input order. Returns dict of numpy outputs."""
        with torch.cuda.stream(self.stream):
            for name, a in zip(self.inputs, arrays):
                t = torch.as_tensor(a) if isinstance(a, np.ndarray) else a
                self.buffers[name].copy_(t.to(self.buffers[name].dtype), non_blocking=True)
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return {n: self.buffers[n].float().cpu().numpy() for n in self.outputs}
