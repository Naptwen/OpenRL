import pyopencl as cl
import pyopencl.array
import numpy as np

# gpu setting for device
platforms = cl.get_platforms()
ctx = cl.Context(dev_type=cl.device_type.GPU,
                 properties=[(cl.context_properties.PLATFORM, platforms[0])])
gpu_queue = cl.CommandQueue(ctx)
# dir_matmul
prg1 = cl.Program(ctx, """
      __kernel void dir_matmul(
      __global const double *A, __global const double *B, __global double *C)
      {
          int index = get_global_id(1) * get_global_size(0) + get_global_id(0);
          C[index] = A[index] * B[index];
      }
    """).build()
# scalar_matmul
prg2 = cl.Program(ctx, """
      __kernel void scalar_matmul(
      const double N, __global const double *A, __global double *C)
      {
          int index = get_global_id(1) * get_global_size(0) + get_global_id(0);; 
          C[index] = N * A[index];
      }
  """).build()
# magic_mul
prg3 = cl.Program(ctx, """
    __kernel void magic_mul(
    const int nB,
    __global const double *A, __global const double *B, __global double *C)
    {
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);;
        int i = index / nB;
        int j = index % nB; 
        C[index] = A[i] * B[j];
    }
    """).build()
# direct_sum
prg4 = cl.Program(ctx, """
      __kernel void direct_sum(
      __global const double *A, __global const double *B, __global double *C)
      {
          int index = get_global_id(1) * get_global_size(0) + get_global_id(0);; 
          C[index] = A[index] + B[index];
      }
  """).build()
# direct_sub
prg5 = cl.Program(ctx, """
      __kernel void direct_sub(
      __global const double *A, __global const double *B, __global double *C)
      {
          int index = get_global_id(1) * get_global_size(0) + get_global_id(0);; 
          C[index] = A[index] - B[index];
      }
  """).build()
# matmul
prg6 = cl.Program(ctx, """
    __kernel void matmul(
    const int Na, const int Nb,
    __global const double *A, __global const double *B, __global double *C)
    {
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);
        int row = index / Nb;
        int col = index % Nb;
        double val = 0.0f;
        for (int t =0; t < Na; t++)
        {
            val += A[row * Na + t] * B[Nb * t + col];
        }
        C[index] = val;
    }
""").build()
# leakReLU
prg7 = cl.Program(ctx, """
    __kernel void leakReLU(
    __global const double *A, __global double *C)
    {
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);;
        C[index] = (0.3*A[index] > A[index])?0.3*A[index]:A[index];
    }
""").build()
# parametricReLU
prg8 = cl.Program(ctx, """
    __kernel void parametricReLU(
    __global const double *A, __global double *C)
    {
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);;
        C[index] = (-0.3*A[index] > A[index])?-0.3*A[index]:A[index];
    }
""").build()
# ReLU
prg9 = cl.Program(ctx, """
    __kernel void ReLU(
    __global const double *A, __global double *C)
    {
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);;
        C[index] = (0 > A[index])?0:A[index];
    }
""").build()
# gr_leakReLU
prg10 = cl.Program(ctx, """
    __kernel void gr_leakReLU(
    __global const double *A, __global double *C)
    {
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);;
        C[index] = (A[index] > 0)? A[index] : 0.3;
    }
""").build()
# gr_parametricReLU
prg11 = cl.Program(ctx, """
    __kernel void gr_parametricReLU(
    __global const double *A, __global double *C)
    {
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);;
        C[index] = (A[index] > 0)? A[index] :-0.3;
    }
""").build()
# gr_ReLU
prg12 = cl.Program(ctx, """
    __kernel void gr_ReLU(
    __global const double *A, __global double *C)
    {
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);;
        C[index] = (A[index] > 0)? A[index] : 0;
    }
""").build()
# loss_fun
prg13 = cl.Program(ctx, """
    __kernel void loss_fn(
    __global const double *A, __global const double *B, __global double *C)
    {
        int index = get_global_id(1) * get_global_size(0) + get_global_id(0);
        C[index]  = -2 * (A[index] - B[index]);
    }
""").build()

def cl_direct_product(A, B, mA, nA):
    C = np.empty(mA * nA, dtype=np.double)
    A = A.astype('double')
    B = B.astype('double')
    Ad = cl.array.to_device(gpu_queue, A)
    Bd = cl.array.to_device(gpu_queue, B)
    Cd = cl.array.to_device(gpu_queue, C)
    prg1.dir_matmul(gpu_queue, (mA , nA), None,
               Ad.data, Bd.data, Cd.data)
    return Cd.get()

def cl_scalar_mul(A, B, mB, nB):
    C = np.empty(mB * nB, dtype=np.double)
    B = B.astype('double')
    Bd = cl.array.to_device(gpu_queue, B)
    Cd = cl.array.to_device(gpu_queue, C)
    prg2.scalar_matmul(gpu_queue, (nB , mB), None,
                       np.double(A), Bd.data, Cd.data)
    return Cd.get()

def cl_magic_mul(A, B, nA, nB):
    C = np.empty(nB * nA, dtype=np.double)
    A = A.astype('double')
    B = B.astype('double')
    Ad = cl.array.to_device(gpu_queue, A)
    Bd = cl.array.to_device(gpu_queue, B)
    Cd = cl.array.to_device(gpu_queue, C)
    prg3.magic_mul(gpu_queue, (nA, nB), None,
                np.int32(nB), Ad.data, Bd.data, Cd.data)
    return Cd.get()

def cl_direct_sum(A, B, mA, nA):
    A = A.astype('double')
    B = B.astype('double')
    C = np.empty(mA * nA, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Bd = cl.array.to_device(gpu_queue, B)
    Cd = cl.array.to_device(gpu_queue, C)
    prg4.direct_sum(gpu_queue, (mA, nA), None,
               Ad.data, Bd.data, Cd.data)
    return Cd.get()

def cl_direct_sub(A, B, mA, nA):
    A = A.astype('double')
    B = B.astype('double')
    C = np.empty(mA * nA, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Bd = cl.array.to_device(gpu_queue, B)
    Cd = cl.array.to_device(gpu_queue, C)
    prg5.direct_sub(gpu_queue, (mA, nA), None,
               Ad.data, Bd.data, Cd.data)
    return Cd.get()

def cl_matmul(A, B, mA, nA, nB):
    A = A.astype('double')
    B = B.astype('double')
    C = np.empty(mA * nB, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Bd = cl.array.to_device(gpu_queue, B)
    Cd = cl.array.to_device(gpu_queue, C)
    prg6.matmul(gpu_queue, (mA, nB), None,
               np.int32(nA), np.int32(nB),
               Ad.data, Bd.data, Cd.data)
    return Cd.get()

def cl_leakReLU(A,mA,nA):
    A = A.astype('double')
    C = np.empty(mA * nA, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Cd = cl.array.to_device(gpu_queue, C)
    prg7.leakReLU(gpu_queue, (mA, nA), None, Ad.data, Cd.data)
    return Cd.get()

def cl_parametricReLU(A,mA,nA):
    A = A.astype('double')
    C = np.empty(mA * nA, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Cd = cl.array.to_device(gpu_queue, C)
    prg8.parametricReLU(gpu_queue, (mA, nA), None, Ad.data, Cd.data)
    return Cd.get()

def cl_ReLU(A,mA,nA):
    A = A.astype('double')
    C = np.empty(mA * nA, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Cd = cl.array.to_device(gpu_queue, C)
    prg9.ReLU(gpu_queue, (mA, nA), None, Ad.data, Cd.data)
    return Cd.get()

def gr_cl_leakReLU(A,mA,nA):
    A = A.astype('double')
    C = np.empty(mA * nA, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Cd = cl.array.to_device(gpu_queue, C)
    prg10.gr_leakReLU(gpu_queue, (mA, nA), None, Ad.data, Cd.data)
    return Cd.get()

def gr_cl_parametricReLU(A,mA,nA):
    A = A.astype('double')
    C = np.empty(mA * nA, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Cd = cl.array.to_device(gpu_queue, C)
    prg11.gr_parametricReLU(gpu_queue, (mA, nA), None, Ad.data, Cd.data)
    return Cd.get()

def gr_cl_ReLU(A,mA,nA):
    A = A.astype('double')
    C = np.empty(mA * nA, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Cd = cl.array.to_device(gpu_queue, C)
    prg12.gr_ReLU(gpu_queue, (mA, nA), None, Ad.data, Cd.data)
    return Cd.get()

def cl_loss(A, B, mA,nA):
    A = A.astype('double')
    B = B.astype('double')
    C = np.empty(mA * nA, dtype=np.double)
    Ad = cl.array.to_device(gpu_queue, A)
    Bd = cl.array.to_device(gpu_queue, B)
    Cd = cl.array.to_device(gpu_queue, C)
    prg13.loss_fn(gpu_queue, (mA, nA), None, Ad.data, Bd.data, Cd.data)
    return Cd.get()