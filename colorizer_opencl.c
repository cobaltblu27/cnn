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

int align(int num, int bound){ 
    return bound * ((num + bound - 1) / bound);
}

/*
 * convolution layer
 * in : (c, h, w)
 * out : (k, h / stride, w / stride)
 * weight : (k, c, 3, 3)
 * bias : (k)
 */
static void conv(cl_mem &in_buff, cl_mem &out_buff,
            cl_mem *weight_buff, cl_mem *bias_buff,
            int H, int W, int K, int C, int stride,
            int act_func_type, int CHW,
        ){
    int Hout = H / stride, Wout = W / stride;
    
    //break down matrix into 16x16 matrix
    //output matrix will be Wout x Hout
    int Wout_align = align(Wout, 16);
    int Hout_align = align(Hout, 16);
    size_t global_size[3] = {K, Wout_align, Hout_align};    
    size_t local_size[3] = {1, 16, 16};

   
    //write to buffer
    err = clEnqueueWriteBuffer(queue, in_buff, CL_TRUE, 0, sizeof(float) H * W * C, in, 0, NULL, NULL);
    CHECK_ERROR(err);

    out_buff = clCreateBuffer(context, 0, sizeof(float) * K * Wout * Hout, NULL, &err);
    CHECK_ERROR(err);

    //pass arguements
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buff);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buff);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight_buff);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias_buff);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &H);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &W);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &K);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &C);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 8, sizeof(int), &stride);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 9, sizeof(int), &act_func_type);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 10, sizeof(int), &CHW);
    CHECK_ERROR(err);

    //run kernel
    err = clEnqueueNDRangeKernel(
            queue, kernel, 3,
            NULL, global_size, local_size,
            0, NULL, NULL
            );
    CHECK_ERROR(err);
}

cm_mem buffer_init(float *in, int H, int W){
    //create buffer
    //TODO
    in_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * C * W * H, NULL, &err);
    CHECK_ERROR(err);
    return in_buff;
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
 * relu : 
 *   out = fmaxf(in, 0);
 * sigmoid :
 *   out = 1 / (1 + expf(-in));
 */

#define FILTER_SIZE 3

cl_mem create_network_buffer(int size, float **network){
    cl_uint err;
    cl_mem buff = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(float) * size, NULL, &err);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, buff, CL_FALSE, 0,
            sizeof(float) * size, *network, 0, NULL, NULL);
    CHECK_ERROR(err);
    *network += size;
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


    cl_mem ll_conv1_w = create_network_buffer(64 * 1 * 3 * 3, &network);
    cl_mem ll_conv1_b = create_network_buffer(64, &network);
    cl_mem ll_conv2_w =  create_network_buffer(128 * 64 * 3 * 3, &network);
    cl_mem ll_conv2_b =  create_network_buffer(128, &network);
    cl_mem ll_conv3_w =  create_network_buffer(128 * 128 * 3 * 3, &network);
    cl_mem ll_conv3_b =  create_network_buffer(128, &network);
    cl_mem ll_conv4_w = create_network_buffer(256 * 128 * 3 * 3, &network);
    cl_mem ll_conv4_b = create_network_buffer(256, &network); 
    cl_mem ll_conv5_w = create_network_buffer(256 * 256 * 3 * 3, &network);
    cl_mem ll_conv5_b = create_network_buffer(256, &network); 
    cl_mem ll_conv6_w = create_network_buffer(512 * 256 * 3 * 3, &network); 
    cl_mem ll_conv6_b = create_network_buffer(512, &network); 
    cl_mem ml_conv1_w = create_network_buffer(512 * 512 * 3 * 3, &network); 
    cl_mem ml_conv1_b = create_network_buffer(512, &network);
    cl_mem ml_conv2_w = create_network_buffer(256 * 512 * 3 * 3, &network); 
    cl_mem ml_conv2_b = create_network_buffer(256, &network); 
    cl_mem gf_conv1_w = create_network_buffer(512 * 512 * 3 * 3, &network);
    cl_mem gf_conv1_b = create_network_buffer(512, &network); 
    cl_mem gf_conv2_w = create_network_buffer(512 * 512 * 3 * 3, &network); 
    cl_mem gf_conv2_b = create_network_buffer(512, &network); 
    cl_mem gf_conv3_w = create_network_buffer(512 * 512 * 3 * 3, &network); 
    cl_mem gf_conv3_b = create_network_buffer(512, &network); 
    cl_mem gf_conv4_w = create_network_buffer(512 * 512 * 3 * 3, &network);
    cl_mem gf_conv4_b = create_network_buffer(512, &network); 
    cl_mem gf_fc1_w = create_network_buffer(1024 * 25088, &network);
    cl_mem gf_fc1_b = create_network_buffer(1024, &network);
    cl_mem gf_fc2_w = create_network_buffer(512 * 1024, &network);
    cl_mem gf_fc2_b = create_network_buffer(512, &network);
    cl_mem gf_fc3_w = create_network_buffer(256 * 512, &network);
    cl_mem gf_fc3_b = create_network_buffer(256, &network);
    cl_mem co_conv1_w = create_network_buffer(256 * 512 * 3 * 3, &network);
    cl_mem co_conv1_b = create_network_buffer(256, &network);
    cl_mem co_conv2_w = create_network_buffer(128 * 256 * 3 * 3, &network);
    cl_mem co_conv2_b = create_network_buffer(128, &network);
    cl_mem co_conv3_w = create_network_buffer(64 * 128 * 3 * 3, &network);
    cl_mem co_conv3_b = create_network_buffer(64, &network);
    cl_mem co_conv4_w = create_network_buffer(64 * 64 * 3 * 3, &network);
    cl_mem co_conv4_b = create_network_buffer(64, &network);
    cl_mem co_conv5_w = create_network_buffer(32 * 64 * 3 * 3, &network);
    cl_mem co_conv5_b = create_network_buffer(32, &network);
    cl_mem co_conv6_w = create_network_buffer(2 * 32 * 3 * 3, &network);
    cl_mem co_conv6_b = create_network_buffer(2, &network);

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
    co_fm4 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    co_fm5 = (float*)malloc(64 * 56 * 56 * sizeof(float));
    co_fm6 = (float*)malloc(64 * 112 * 112 * sizeof(float));
    co_fm7 = (float*)malloc(32 * 112 * 112 * sizeof(float));

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

