#define RELU 0
#define SIGMOID 1
#define NONE 2

/*
 * Convolution Layer
 * in : (C, H, W)
 * out : (K, H / stride, W / stride)
 * weight : (K, C, 3, 3)
 * bias : (K)
 */
__kernel void conv(
        __global float *in,
        __global float *out,
        __global float *weight, 
        __global float *bias,
        int H, int W, int K, int C,
        int act_func_type
        ){
    
    activation(act_func_type, out, K, Hout, Wout);
}

//input size: C, weight size: C * K, output size: K
__kernel void fc(
        __global float *in,
        __global float *out,
        __global float *weight, 
        __global float *bias,
        int K, int C, int act_func_type
        ){
    k = get_global_id(0);
    if(k < K){
        float sum = bias[k];
        for(int c = 0; c < C; c++)
            sum += weight[C * k + c] * in[c];
        out[k] = sum;
    }
    activation(act_func_type, out, K, 1, 1);
}

__kernel void activation(int type, __local float *inout,
        int size_k, int size_h, int size_w){
    int k = get_global_id(0);
    int h = get_global_id(1);
    int w = get_global_id(2);
    int index = k * size_h * size_w + h * size_w + w;
    if(k < size_k && h < size_h && w < size_w)
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
