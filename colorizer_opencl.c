#include <CL/cl.h>
#include "colorizer.h"

#define RELU 0
#define SIGMOID 1
/*
 * TODO
 * Define global variables here. For example,
 * cl_platform_id platform;
 */
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_uint err;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;

static void conv(float *in, float *out, float *weight, float *bias,
            int H, int W, int K, int C, int stride,
            int act_func_type, int CHW
        ){
    int HOUT = H / stride, WOUT = W / stride;
    //TODO: implement with opencl
    
    //make vector and add
    cl_mem buffA, buffB, buffC;

    //break down matrix into 16x16 matrix
    int A_row_align = ALIGN(ROW_A);
    int A_col_align = ALIGN(COL_A);
    int B_col_align = ALIGN(COL_B);
    size_t global_size[2] = {A_row_align, B_col_align};    
    size_t local_size[2] = {16, 16};

    //create buffer 
    buffA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ROW_A * COL_A, NULL, &err);
    CHECK_ERROR(err);
    buffB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * COL_A * COL_B, NULL, &err);
    CHECK_ERROR(err);
    
    //write to buffer
    err = clEnqueueWriteBuffer(queue, buffA, CL_TRUE, 0, sizeof(float) * COL_A * ROW_A, A, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, buffB, CL_TRUE, 0, sizeof(float) * COL_A * COL_B, B, 0, NULL, NULL);
    CHECK_ERROR(err);


    buffC = clCreateBuffer(context, 0, sizeof(float) * ROW_A * COL_B, NULL, &err);
    CHECK_ERROR(err);

    //pass arguements
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffA);
    CHECK_ERROR(err);

    //run kernel
    err = clEnqueueNDRangeKernel(
            queue, kernel, 2,
            NULL, global_size, local_size,
            0, NULL, NULL
            );
    CHECK_ERROR(err);

    //pass output
    err = clEnqueueReadBuffer(queue, buffC, CL_TRUE, 0, sizeof(float) * ROW_A * COL_B, C, 0, NULL, NULL);
    CHECK_ERROR(err);

}

static void fc(float *in, float *out, float *weight,
            float *bias, int K, int C, int act_func_type, int CHW
        ){
    //TODO: implement with opencl

    err = clEnqueueNDRangeKernel(
            queue, kernel, 2,
            NULL, global_size, local_size,
            0, NULL, NULL
            );
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, buffC, CL_TRUE, 0, sizeof(float) * ROW_A * COL_B, C, 0, NULL, NULL);
    CHECK_ERROR(err);

}

static void fuse(float *ml, float *fg, float *out){
    //TODO
}

static void upsample(float *in, float *out, int H, int W, int C){
    //TODO
}
/*
// ReLU (in-place)
// inout : (C, H, W)

static void relu(float *inout, int CHW) {
    for (int chw = 0; chw < CHW; ++chw) {
        inout[chw] = fmaxf(inout[chw], 0);
    }
}


// Sigmoid (in-place)
// inout : (C, H, W)

static void sigmoid(float *inout, int CHW) {
    for (int chw = 0; chw < CHW; ++chw) {
        inout[chw] = 1 / (1 + expf(-inout[chw]));
    }
}

// ml : (256, 28, 28)
// gf : (256)
// out : (512, 28, 28)

static void fuse(float *ml, float *gf, float *out) {
    for (int k = 0; k < 256; ++k) {
        for (int h = 0; h < 28; ++h) {
            for (int w = 0; w < 28; ++w) {
                out[k * 28 * 28 + h * 28 + w] = ml[k * 28 * 28 + h * 28 + w];
            }
        }
    }
    for (int k = 256; k < 512; ++k) {
        for (int h = 0; h < 28; ++h) {
            for (int w = 0; w < 28; ++w) {
                out[k * 28 * 28 + h * 28 + w] = gf[k - 256];
            }
        }
    }
}


// in : (C, H, W)
// out : (C, H * 2, W * 2)
static void upsample(float *in, float *out, int H, int W, int C) {
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float t = in[c * H * W + h * W + w];
                out[c * H * W * 4 + (2 * h + 0) * W * 2 + (2 * w + 0)] = t;
                out[c * H * W * 4 + (2 * h + 0) * W * 2 + (2 * w + 1)] = t;
                out[c * H * W * 4 + (2 * h + 1) * W * 2 + (2 * w + 0)] = t;
                out[c * H * W * 4 + (2 * h + 1) * W * 2 + (2 * w + 1)] = t;
            }
        }
    }
}
*/

/*
 * relu : 
 *   out = fmaxf(in, 0);
 * sigmoid :
 *   out = 1 / (1 + expf(-in));
 */
void colorizer_init() {
    /*
     * TODO
     * Initialize OpenCL objects as global variables. For example,
     * clGetPlatformIDs(1, &platform, NULL);
     */
    char *source_code;
    size_t len;
    int i;

    //get platform and device IDs
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);
    
    //make queue, program and build
    source_code = get_source_code("kernel.cl", &len);
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);
    program = clCreateProgramWithSource(context, 1, (const char**)&source_code, NULL, &err);
    CHECK_ERROR(err);
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -Werror", NULL, NULL);

    //check compile error
    if(err == CL_BUILD_PROGRAM_FAILURE){
        size_t log_size;
        char *log;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        log = (char *)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        printf("Compile error:\n%s\n", log);
        free(log);
    }

    //create kernel
    kernel = clCreateKernel(program, "colorize", &err);
    CHECK_ERROR(err);

}

void colorizer(int nimg, float *network, float *inputs, float *outputs) {
    /*
     * TODO
     * Implement here.
     * See "colorizer_seq.c" if you don't know what to do.
     */

    // ll = Low-Level Feature Network
    // ml = Mid-Level Feature Network
    // gf = Global Feature Network
    // co = Colorization Network
    // w = weight, b = bias

    // split network into each layer's parameter

    /* 
     * network += 64 * 1 * 3 * 3 means 
     * K = 64 (number of input channel, number of input sets)
     * C = 1 (number of output channel, number of weight sets)
     * and 3 by 3 filter
     * H and W stands for height and width
     */
    float *ll_conv1_w = network; network += 64 * 1 * 3 * 3;
    float *ll_conv1_b = network; network += 64;
    float *ll_conv2_w = network; network += 128 * 64 * 3 * 3;
    float *ll_conv2_b = network; network += 128;
    float *ll_conv3_w = network; network += 128 * 128 * 3 * 3;
    float *ll_conv3_b = network; network += 128;
    float *ll_conv4_w = network; network += 256 * 128 * 3 * 3;
    float *ll_conv4_b = network; network += 256;
    float *ll_conv5_w = network; network += 256 * 256 * 3 * 3;
    float *ll_conv5_b = network; network += 256;
    float *ll_conv6_w = network; network += 512 * 256 * 3 * 3;
    float *ll_conv6_b = network; network += 512;
    float *ml_conv1_w = network; network += 512 * 512 * 3 * 3;
    float *ml_conv1_b = network; network += 512;
    float *ml_conv2_w = network; network += 256 * 512 * 3 * 3;
    float *ml_conv2_b = network; network += 256;
    float *gf_conv1_w = network; network += 512 * 512 * 3 * 3;
    float *gf_conv1_b = network; network += 512;
    float *gf_conv2_w = network; network += 512 * 512 * 3 * 3;
    float *gf_conv2_b = network; network += 512;
    float *gf_conv3_w = network; network += 512 * 512 * 3 * 3;
    float *gf_conv3_b = network; network += 512;
    float *gf_conv4_w = network; network += 512 * 512 * 3 * 3;
    float *gf_conv4_b = network; network += 512;
    float *gf_fc1_w = network; network += 1024 * 25088;
    float *gf_fc1_b = network; network += 1024;
    float *gf_fc2_w = network; network += 512 * 1024;
    float *gf_fc2_b = network; network += 512;
    float *gf_fc3_w = network; network += 256 * 512;
    float *gf_fc3_b = network; network += 256;
    float *co_conv1_w = network; network += 256 * 512 * 3 * 3;
    float *co_conv1_b = network; network += 256;
    float *co_conv2_w = network; network += 128 * 256 * 3 * 3;
    float *co_conv2_b = network; network += 128;
    float *co_conv3_w = network; network += 64 * 128 * 3 * 3;
    float *co_conv3_b = network; network += 64;
    float *co_conv4_w = network; network += 64 * 64 * 3 * 3;
    float *co_conv4_b = network; network += 64;
    float *co_conv5_w = network; network += 32 * 64 * 3 * 3;
    float *co_conv5_b = network; network += 32;
    float *co_conv6_w = network; network += 2 * 32 * 3 * 3;
    float *co_conv6_b = network; network += 2;

    // intermediate buffer for feature maps
    float *ll_fm1 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    float *ll_fm2 = (float*)malloc(128 * 112 * 112 * sizeof(float));
    float *ll_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    float *ll_fm4 = (float*)malloc(256 * 56 * 56 * sizeof(float));
    float *ll_fm5 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    float *ll_fm6 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    float *ml_fm1 = (float*)malloc(512 * 28 * 28 * sizeof(float));
    float *ml_fm2 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    float *gf_fm1 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    float *gf_fm2 = (float*)malloc(512 * 14 * 14 * sizeof(float));
    float *gf_fm3 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    float *gf_fm4 = (float*)malloc(512 * 7 * 7 * sizeof(float));
    float *gf_fm5 = (float*)malloc(1024 * sizeof(float));
    float *gf_fm6 = (float*)malloc(512 * sizeof(float));
    float *gf_fm7 = (float*)malloc(256 * sizeof(float));
    float *ml_gf_fused_fm = (float*)malloc(512 * 28 * 28 * sizeof(float));
    float *co_fm1 = (float*)malloc(256 * 28 * 28 * sizeof(float));
    float *co_fm2 = (float*)malloc(128 * 28 * 28 * sizeof(float));
    float *co_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
    float *co_fm4 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    float *co_fm5 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    float *co_fm6 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    float *co_fm7 = (float*)malloc(32 * 112 * 112 * sizeof(float));

    // run network for each image
    for (int n = 0; n < nimg; ++n) {
        float *input = inputs + n * 224 * 224;
        float *output = outputs + n * 2 * 112 * 112;
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

        conv(ll_fm6, ml_fm1, ml_conv1_w, ml_conv1_b, 28, 28, 512, 512, 1);
        relu(ml_fm1, 512 * 28 * 28);
        conv(ml_fm1, ml_fm2, ml_conv2_w, ml_conv2_b, 28, 28, 256, 512, 1);
        relu(ml_fm2, 256 * 28 * 28);

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

        fuse(ml_fm2, gf_fm7, ml_gf_fused_fm);

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
        sigmoid(output, 2 * 112 * 112);
    }

    free(source_code);
    clReleaseMemObject(buffA);
    clReleaseMemObject(buffB);
    clReleaseMemObject(buffC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

