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

__kernel void ReLU(__local float *inout, int CHW){
    //TODO: out = fmaxf(in, 0);
}

__kernel void sigmoid(__local float *inout, int CHW){
    //TODO: out = 1 / (1 + exp(-in));
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
