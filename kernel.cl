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

__kernel void activation(int type, __global float *inout,
        int size_k, int size_h, int size_w){
    int k = get_global_id(0);
    int h = get_global_id(1);
    int w = get_global_id(2);
    int index = k * size_h * size_w + h * size_w + w;
    if(k < size_k && h < size_h && w < size_w){
        if(type == RELU){
            inout[index] = fmax(inout[index], 0);
        }else if(type == SIGMOID){
            inout[index] = 1 / (1 + exp(-inout[index]));
        }
    }
}

__kernel void conv(
        __global float *in,
        __global float *out,
        __global float *weight, 
        __global float *bias,
        int H, int W, int K, int C, int stride, 
        int act_func_type
        ){
    
	int HOUT = H / stride, WOUT = W / stride;
	int CHW = C*HOUT*WOUT;
	int wout = get_global_id(0);
	int hout = get_global_id(1);
	int k = get_global_id(2);
	
	if( k < K && hout < HOUT && wout < WOUT) {
		float sum = bias[k];
		for (int c = 0; c < C; ++c){
			for (int r = 0; r < 3; ++r) {
				for (int s = 0; s < 3; ++s) {
					// calculate position in input image
                    int h = hout * stride + r - 1;
                    int w = wout * stride + s - 1;
                    if (h < 0 || h >= H || w < 0 || w >= W) {
                     // out of bound, do nothing
                    } else {
                        sum += in[c * H * W + h * W + w] * weight[k * C * 3 * 3 + c * 3 * 3 + r * 3 + s];
					}
		        }
            }
	    }
        out[k * HOUT * WOUT + hout * WOUT + wout] = sum;
    }
    activation(act_func_type, out, K, HOUT, WOUT);
}

//input size: C, weight size: C * K, output size: K
__kernel void fc(
        __global float *in,
        __global float *out,
        __global float *weight, 
        __global float *bias,
        int K, int C, int act_func_type
        ){
    int k = get_global_id(0);
    if(k < K){
        float sum = bias[k];
        for(int c = 0; c < C; c++)
            sum += weight[C * k + c] * in[c];
        out[k] = sum;
    }
    activation(act_func_type, out, K, 1, 1);
}
/*
 * workgroup: (256, 28, 28)
 */
__kernel void fuse(
        __global float *ml,
        __global float *gf,
        __global float *out
        ){
    int k = get_global_id(0);
    int h = get_global_id(1);
    int w = get_global_id(2);
    if(k < 256 && h < 28 && w < 28){
        out[k * 28 * 28 + h * 28 + w] = ml[k * 28 * 28 + h * 28 + w];
        out[(k + 256) * 28 * 28 + h * 28 + w] = gf[k];
    }
}
/*
 * in : (C, H, W)
 * out : (C, H * 2, W * 2)
 */
__kernel void up_sample(
        __global float *in,
        __global float *out,
        int H, int W, int C
        ){
    int c = get_global_id(0);
    int h = get_global_id(1);
    int w = get_global_id(2);
    if(c < C && h < H && w < W){
        float t = in[c * H * W + h * W + w];
        out[c * H * W * 4 + (2 * h + 0) * W * 2 + (2 * w + 0)] = t;
        out[c * H * W * 4 + (2 * h + 0) * W * 2 + (2 * w + 1)] = t;
        out[c * H * W * 4 + (2 * h + 1) * W * 2 + (2 * w + 0)] = t;
        out[c * H * W * 4 + (2 * h + 1) * W * 2 + (2 * w + 1)] = t;
    }
}



