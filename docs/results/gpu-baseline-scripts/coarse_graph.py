"""Prototype: enqueue the unchanged regular-grid relaxation sweeps as a CUDA graph."""
import threading
from functools import lru_cache
import warp as wp
from warp_psr.relaxation import graph_kernels, SlicedGraph, relax_scheduled as original

@lru_cache(maxsize=32)
def graph(dtype,device,stream,iters,forward,capacity):
    state_type,bind,update,_=graph_kernels(dtype)
    descriptor=wp.empty(1,dtype=state_type,device=device)
    command=wp.launch(update,capacity,inputs=[descriptor,0],device=device,record_cmd=True)
    wp.launch(bind,1,inputs=[state_type(),descriptor],device=device,record_cmd=True)
    with wp.ScopedCapture(device=device) as capture:
        for _ in range(iters):
            for group in (range(8) if forward else range(7,-1,-1)):
                command.set_param_at_index(1,group);command.launch()
    return SlicedGraph(descriptor,capture.graph,wp.get_stream(device),threading.Lock())

def relax_scheduled(A,b,x,schedule,iters,forward=True,z=-1,command=None):
    if z>=0 or not x.device.is_cuda or x.device.is_capturing or iters<1:
        return original(A,b,x,schedule,iters,forward,z,command)
    capacity=1 << max(0,schedule.maximum-1).bit_length()
    cached=graph(b.dtype,str(x.device),wp.get_stream(x.device).cuda_stream,iters,forward,capacity)
    state_type,bind,_,_=graph_kernels(b.dtype)
    state=state_type()
    state.rows,state.group_offsets=schedule.rows,schedule.offsets
    state.offsets,state.columns,state.values=A.offsets,A.columns,A.values
    state.b,state.x=b,x
    with cached.lock:
        wp.launch(bind,1,inputs=[state,cached.descriptor],device=x.device)
        wp.capture_launch(cached.graph)
