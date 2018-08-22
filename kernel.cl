float *ll_conv1_w;
float *ll_conv1_b;
float *ll_conv2_w;
float *ll_conv2_b;
float *ll_conv3_w;
float *ll_conv3_b;
float *ll_conv4_w;
float *ll_conv4_b;
float *ll_conv5_w;
float *ll_conv5_b;
float *ll_conv6_w;
float *ll_conv6_b;
float *ml_conv1_w;
float *ml_conv1_b;
float *ml_conv2_w;
float *ml_conv2_b;
float *gf_conv1_w;
float *gf_conv1_b;
float *gf_conv2_w;
float *gf_conv2_b;
float *gf_conv3_w;
float *gf_conv3_b;
float *gf_conv4_w;
float *gf_conv4_b;
float *gf_fc1_w;
float *gf_fc1_b;
float *gf_fc2_w;
float *gf_fc2_b;
float *gf_fc3_w;
float *gf_fc3_b;
float *co_conv1_w;
float *co_conv1_b;
float *co_conv2_w;
float *co_conv2_b;
float *co_conv3_w;
float *co_conv3_b;
float *co_conv4_w;
float *co_conv4_b;
float *co_conv5_w;
float *co_conv5_b;
float *co_conv6_w;
float *co_conv6_b;


float *ll_fm1; 
float *ll_fm2;
float *ll_fm3;
float *ll_fm4;
float *ll_fm5;
float *ll_fm6;
float *ml_fm1;
float *ml_fm2;
float *gf_fm1;
float *gf_fm2;
float *gf_fm3;
float *gf_fm4;
float *gf_fm5;
float *gf_fm6;
float *gf_fm7;
float *ml_gf_fused_fm;
float *co_fm1;
float *co_fm2;
float *co_fm3;
float *co_fm4;
float *co_fm5;
float *co_fm6;
float *co_fm7;


__kernel void fuse(
        __global float *ml,
        __global float *gf,
        __global float *out
        ){

}

__kernel void up_sample(
        __global float *in,
        __global float *out,
        int H, int W, int C
        ){

}
__kernel void conv(__local float *in,
        __local float *out,
        __local float *weight,
        __local float *bias,
        int H, int W, int K, int C, int stride,
        int act_func_type, int CHW){
    int HOUT = H / stride, WOUT = W / stride;
    int i, j, k;
    int channel = get_global_id(0); // input layer
    int w_id = get_global_id(1); // width
    int h_id = get_global_id(2); // height
    int w, h;
    
    if(w < WOUT && h < HOUT){
        for(k = 0; k < K; k++){
            int sum = bias[k];
            for(i = 0; i < 3; i++){
                for(j = 0; j < 3; j++){
                    //TODO get coordinates with stride, w and h
                    h = h_id * stride + i - 1;
                    w = w_id * stride + j - 1;
                    if(h >= 0 && h < H && w >= 0 && w < W){
                        sum += in[channel * H * W + h * W + w] 
                            * weight[k * C * 9 + channel * 9 + i * 3 + j];
                    }
                }
            }
            out[k * HOUT * WOUT + hout * WOUT + wout] = sum;
        }
    }
}

// 16 by 16
float ReLU(float *inout, int CHW){
    int index = get_global_id(1)
    return fmaxf(inout, 0);
}

float sigmoid(float inout, int CHW){
    return 1 / (1 + exp(-inout));
}

void low_level(){
    conv(input, ll_fm1, ll_conv1_w, ll_conv1_b, 224, 224, 64, 1, 2);
    relu(ll_fm1, 64 * 112 * 112);
    conv(ll_fm1, ll_fm2, ll_conv2_w, ll_conv2_b, 112, 112, 128, 64, 1);
    relu(ll_fm2, 128 * 112 * 112);
    conv(ll_fm2, ll_fm3, ll_conv3_w, ll_conv3_b, 112, 112, 128, 128, 2);
    relu(ll_fm3, 128 * 56 * 56);
    conv(ll_fm3, ll_fm4, ll_conv4_w, ll_conv4_b, 56, 56, 256, 128, 1);
    relu(ll_fm4, 256 * 56 * 56);
    conv(ll_fm4, ll_fm5, ll_conv5_w, ll_conv5_b, 56, 56, 256, 256, 2);
    relu(ll_fm5, 256 * 28 * 28);
    conv(ll_fm5, ll_fm6, ll_conv6_w, ll_conv6_b, 28, 28, 512, 256, 1);
    relu(ll_fm6, 512 * 28 * 28);
}

void mid_level(){
    conv(ll_fm6, ml_fm1, ml_conv1_w, ml_conv1_b, 28, 28, 512, 512, 1);
    relu(ml_fm1, 512 * 28 * 28);
    conv(ml_fm1, ml_fm2, ml_conv2_w, ml_conv2_b, 28, 28, 256, 512, 1);
    relu(ml_fm2, 256 * 28 * 28);
}

void global_feature(){
    conv(ll_fm6, gf_fm1, gf_conv1_w, gf_conv1_b, 28, 28, 512, 512, 2);
    relu(gf_fm1, 512 * 14 * 14);
    conv(gf_fm1, gf_fm2, gf_conv2_w, gf_conv2_b, 14, 14, 512, 512, 1);
    relu(gf_fm2, 512 * 14 * 14);
    conv(gf_fm2, gf_fm3, gf_conv3_w, gf_conv3_b, 14, 14, 512, 512, 2);
    relu(gf_fm3, 512 * 7 * 7);
    conv(gf_fm3, gf_fm4, gf_conv4_w, gf_conv4_b, 7, 7, 512, 512, 1);
    relu(gf_fm4, 512 * 7 * 7);
    fc(gf_fm4, gf_fm5, gf_fc1_w, gf_fc1_b, 1024, 25088);
    relu(gf_fm5, 1024);
    fc(gf_fm5, gf_fm6, gf_fc2_w, gf_fc2_b, 512, 1024);
    relu(gf_fm6, 512);
    fc(gf_fm6, gf_fm7, gf_fc3_w, gf_fc3_b, 256, 512);
    relu(gf_fm7, 256);
}

void co_net(){
    conv(ml_gf_fused_fm, co_fm1, co_conv1_w, co_conv1_b, 28, 28, 256, 512, 1);
    relu(co_fm1, 256 * 28 * 28);
    conv(co_fm1, co_fm2, co_conv2_w, co_conv2_b, 28, 28, 128, 256, 1);
    relu(co_fm2, 128 * 28 * 28);
    upsample(co_fm2, co_fm3, 28, 28, 128);
    conv(co_fm3, co_fm4, co_conv3_w, co_conv3_b, 56, 56, 64, 128, 1);
    relu(co_fm4, 64 * 56 * 56);
    conv(co_fm4, co_fm5, co_conv4_w, co_conv4_b, 56, 56, 64, 64, 1);
    relu(co_fm5, 64 * 56 * 56);
    upsample(co_fm5, co_fm6, 56, 56, 64);
    conv(co_fm6, co_fm7, co_conv5_w, co_conv5_b, 112, 112, 32, 64, 1);
    relu(co_fm7, 32 * 112 * 112);
    conv(co_fm7, output, co_conv6_w, co_conv6_b, 112, 112, 2, 32, 1);
}

__kernel void colorize(
        __global float *inputs,
        __global float *outputs,
        int nimg
        ){

    //float *input = inputs + n * 224 * 224;
    //float *output = outputs + n * 2 * 112 * 112;

    low_level();
    mid_level();
    global_feature();
    
    fuse(ml_fm2, gf_fm7, ml_gf_fused_fm);
    
    co_net();

    sigmoid(output, 2 * 112 * 112);
}

__kernel void fc(
        __global float *in,
        __global float *out,
        __global float *weight, 
        __global float *bias,
        int K, int C, int act_func_type, int CHW
        ){

}


//initialize network
__kernel void init(__global float *network){
    *ll_conv1_w = network; network += 64 * 1 * 3 * 3;
    *ll_conv1_b = network; network += 64;
    *ll_conv2_w = network; network += 128 * 64 * 3 * 3;
    *ll_conv2_b = network; network += 128;
    *ll_conv3_w = network; network += 128 * 128 * 3 * 3;
    *ll_conv3_b = network; network += 128;
    *ll_conv4_w = network; network += 256 * 128 * 3 * 3;
    *ll_conv4_b = network; network += 256;
    *ll_conv5_w = network; network += 256 * 256 * 3 * 3;
    *ll_conv5_b = network; network += 256;
    *ll_conv6_w = network; network += 512 * 256 * 3 * 3;
    *ll_conv6_b = network; network += 512;
    *ml_conv1_w = network; network += 512 * 512 * 3 * 3;
    *ml_conv1_b = network; network += 512;
    *ml_conv2_w = network; network += 256 * 512 * 3 * 3;
    *ml_conv2_b = network; network += 256;
    *gf_conv1_w = network; network += 512 * 512 * 3 * 3;
    *gf_conv1_b = network; network += 512;
    *gf_conv2_w = network; network += 512 * 512 * 3 * 3;
    *gf_conv2_b = network; network += 512;
    *gf_conv3_w = network; network += 512 * 512 * 3 * 3;
    *gf_conv3_b = network; network += 512;
    *gf_conv4_w = network; network += 512 * 512 * 3 * 3;
    *gf_conv4_b = network; network += 512;
    *gf_fc1_w = network; network += 1024 * 25088;
    *gf_fc1_b = network; network += 1024;
    *gf_fc2_w = network; network += 512 * 1024;
    *gf_fc2_b = network; network += 512;
    *gf_fc3_w = network; network += 256 * 512;
    *gf_fc3_b = network; network += 256;
    *co_conv1_w = network; network += 256 * 512 * 3 * 3;
    *co_conv1_b = network; network += 256;
    *co_conv2_w = network; network += 128 * 256 * 3 * 3;
    *co_conv2_b = network; network += 128;
    *co_conv3_w = network; network += 64 * 128 * 3 * 3;
    *co_conv3_b = network; network += 64;
    *co_conv4_w = network; network += 64 * 64 * 3 * 3;
    *co_conv4_b = network; network += 64;
    *co_conv5_w = network; network += 32 * 64 * 3 * 3;
    *co_conv5_b = network; network += 32;
    *co_conv6_w = network; network += 2 * 32 * 3 * 3;
    *co_conv6_b = network; network += 2;

    *ll_fm1 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    *ll_fm2 = (float*)malloc(128 * 112 * 112 * sizeof(float));
    *ll_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    *ll_fm4 = (float*)malloc(256 * 56 * 56 * sizeof(float));
    *ll_fm5 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    *ll_fm6 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    *ml_fm1 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    *ml_fm2 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    *gf_fm1 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    *gf_fm2 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    *gf_fm3 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    *gf_fm4 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    *gf_fm5 = (float*)malloc(1024 * sizeof(float));
    *gf_fm6 = (float*)malloc(512 * sizeof(float));
    *gf_fm7 = (float*)malloc(256 * sizeof(float));
    *ml_gf_fused_fm = (float*)malloc(512 * 28 * 28 * sizeof(float));
    *co_fm1 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    *co_fm2 = (float*)malloc(128 * 28 * 28 * sizeof(float));
    *co_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    *co_fm4 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    *co_fm5 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    *co_fm6 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    *co_fm7 = (float*)malloc(32 * 112 * 112 * sizeof(float));
}


