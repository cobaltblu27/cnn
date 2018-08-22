#define RELU 0
#define SIGMOID 1
#define NONE 2

__kernel void conv(
        __global float *in,
        __global float *out,
        __global float *weight, 
        __global float *bias,
        int H, int W, int K, int C,
        int act_func_type, int CHW
        ){

}

__kernel void fc(
        __global float *in,
        __global float *out,
        __global float *weight, 
        __global float *bias,
        int K, int C, int act_func_type, int CHW
        ){

}

__kernel void activation(int type, __local float *inout, int CHW){
    int Z_size = get_global_size(0);
    int X_size = get_global_size(1);
    int Y_size = get_global_size(2);
    int index = get_global_id(0) * Z_size + get_global_id(1) * X_size + get_global_id(2) * Y_size;
    if(index < CHW)
        if(type == RELU)
            inout[index] = fmax(inout[index], 0);
        else if(type == SIGMOID)
            inout[index] = 1 / (1 + exp(-inout[index]));
}

/*** These fuctions might not be used in kernel ***/

 /*
__kernel void fuse(
        __global float *ml,
        __global float *gf,
        __global float *out
        ){

}

__kernal void up_sample(
        __global float *in,
        __global float *out,
        int H, int W, int C
        ){

}
*/
