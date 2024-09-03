//+------------------------------------------------------------------+
//|                                              Sample DLL for MQL4 |
//|                             Copyright 2000-2024, MetaQuotes Ltd. |
//|                                               www.metaquotes.net |
//+------------------------------------------------------------------+
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include "CL/cl.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
//#include <OpenCL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (1024)


struct _CLKernel 
{
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
};
typedef struct _CLKernel* CLKernel;
CLKernel _kernel_neural_forwardpass;
CLKernel _kernel_neural_propoutputs;
CLKernel _kernel_neural_training_backprop;

////////////////////////////////////////////////////////////////////////////////

// Compute kernel which does a forward pass of an input array (weights) to outputs.
//
const char* KernelSource_NeuralForwardPass = "\n" \
"__kernel void square(                                                  \n" \
"   __global double* input,                                             \n" \
"   __global double* output,                                            \n" \
"   --global double input_equity,                                       \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"   {                                                                   \n" \
"       double sum = input[i * 0] * input_equity + input[i * 1];        \n" \
"       output[i] = 1.0 / (1.0 + exp(sum)));                            \n" \
"   }                                                                   \n" \
"}                                                                      \n" \
"\n"; // 1.0 / (1.0 + exp(sum))) is the sigmoid activation function.

/*
for (int i = 0; i < NNET_hidden_count; i++)
{
    double sum = weights[i * 0] * inputEquity + weights[i * 1];
    hiddenValues[i] = sigmoid(sum);
}
// */

const char* KernelSource_NeuralPropOutputs = "\n" \
"__kernel void square(                                                  \n" \
"   __global double* input,                                              \n" \
"   __global double* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"   {                                                                   \n" \
"       output[0] = output[0] + input[i];                               \n" \
"   }                                                                   \n" \
"}                                                                      \n" \
"\n";

/*
double outputValue = 0;
for (int i = 0; i < NNET_hidden_count; i++)
{
  outputValue += hiddenValues[i];
}
// */

const char* KernelSource_NeuralTrainBackprop = "\n" \
"__kernel void square(                                                                          \n" \
"   __global double* input,                                                                     \n" \
"   __global double* output,                                                                    \n" \
"   --global double* hiddenValues,                                                              \n" \
"   --global unsigned int NNET_outputs_count,                                                   \n" \
"   --global double error,                                                                      \n" \
"   --global double* deltaWeights,                                                              \n" \
"   --global double NNET_learningRate,                                                          \n" \
"   --global double NNET_momentum,                                                              \n" \
"   const unsigned int count)                                                                   \n" \
"{                                                                                              \n" \
"   int i = get_global_id(0);                                                                   \n" \
"   if(i < count)                                                                               \n" \
"   {                                                                                           \n" \
"       for (int j = 0; j < NNET_outputs_count; j++)                                            \n" \
"       {                                                                                       \n" \
"           double delta = error * hiddenValues[i] * (1 - hiddenValues[i]);                     \n" \
"           deltaWeights[i*j] = NNET_learningRate * delta + NNET_momentum * deltaWeights[i*j];  \n" \
"           output[i*j] = output[i*j] + deltaWeights[i*j];                                      \n" \
"       }                                                                                       \n" \
"   }                                                                                           \n" \
"}                                                                                              \n" \
"\n";

/*
// Backward pass
for (int i = 0; i < NNET_hidden_count; i++)
{
  for (int j = 0; j < NNET_outputs_count; j++)
  {
     double delta = error * hiddenValues[i] * (1 - hiddenValues[i]);
     deltaWeights[i*j] = NNET_learningRate * delta + NNET_momentum * deltaWeights[i*j];
     weights[i*j] += deltaWeights[i*j];

  }
}
 // */


////////////////////////////////////////////////////////////////////////////////

int exec_kernel_training(CLKernel _kernel, double error, double NNET_learningRate, double NNET_momentum, 
    double* input_weights, double** output_weights, double* hiddenValues, int NNET_outputs_count, double** deltaWeights,
    const int arraysize)
{
    int i = 0;
    unsigned int count = arraysize;

    unsigned int results_count = 1;

    (*output_weights) = (double*)malloc(sizeof(double) * arraysize);

    cl_kernel kernel = _kernel->kernel;
    cl_context context = _kernel->context;
    cl_command_queue commands = _kernel->commands;

    int err;                            // error code returned from api calls
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    cl_mem hidden;                      // device memory used for hiddenValues array
    cl_mem delta_weights;               // device memory used for deltaWeights aray
    /*
        "   __global double* input,                                                                     \n" \
        "   __global double* output,                                                                    \n" \
        "   --global double* hiddenValues,                                                              \n" \
        "   --global unsigned int NNET_outputs_count,                                                   \n" \
        "   --global double error,                                                                      \n" \
        "   --global double* deltaWeights,                                                              \n" \
        "   --global double NNET_learningRate,                                                          \n" \
        "   --global double NNET_momentum,                                                              \n" \
    */
    // Create the input and output arrays and hidden and delta weights in device memory for our calculation

    input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * (count * NNET_outputs_count), NULL, NULL);
    hidden = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * count, NULL, NULL);
    delta_weights = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * (count * NNET_outputs_count), NULL, NULL);

    if (!input || !output || !hidden || !delta_weights)
    {
        printf("Error: Failed to allocate device memory!\n");
        //exit(1);
        return EXIT_FAILURE;
    }

    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, input_weights, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        //exit(1);
        return EXIT_FAILURE;
    }

    err = clEnqueueWriteBuffer(commands, hidden, CL_TRUE, 0, sizeof(float) * count, hiddenValues, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        //exit(1);
        return EXIT_FAILURE;
    }

    err = clEnqueueWriteBuffer(commands, delta_weights, CL_TRUE, 0, sizeof(float) * count, *deltaWeights, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        //exit(1);
        return EXIT_FAILURE;
    }

    /*
    "   __global double* input,                                                                     \n" \
    "   __global double* output,                                                                    \n" \
    "   --global double* hiddenValues,                                                              \n" \
    "   --global unsigned int NNET_outputs_count,                                                   \n" \
    "   --global double error,                                                                      \n" \
    "   --global double* deltaWeights,                                                              \n" \
    "   --global double NNET_learningRate,                                                          \n" \
    "   --global double NNET_momentum,                                                              \n" \
    "   const unsigned int count)                                                                   \n" \
    */

    // Set the arguments to our compute kernel
//
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &hiddenValues);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &NNET_outputs_count);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, _kernel->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(double) * results_count, *output_weights, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    err = clEnqueueReadBuffer(commands, delta_weights, CL_TRUE, 0, sizeof(double) * results_count, *deltaWeights, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    // Release objects.
    clReleaseMemObject(input);
    clReleaseMemObject(output);

    return EXIT_SUCCESS;
}

int exec_kernel_singleoutput(CLKernel _kernel, double* outputValue, double* hiddenValues, const int arraysize)
{
    int i = 0;
    unsigned int count = arraysize;

    unsigned int results_count = 1;

    double* results_array = (double*)malloc(sizeof(double) * 1);
    if (results_array != nullptr)
    {
        results_array[0] = 0.0;
    }

    cl_kernel kernel = _kernel->kernel;
    cl_context context = _kernel->context;
    cl_command_queue commands = _kernel->commands;

    int err;                            // error code returned from api calls
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    // Create the input and output arrays in device memory for our calculation

    input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        //exit(1);
        return EXIT_FAILURE;
    }

    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, hiddenValues, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        //exit(1);
        return EXIT_FAILURE;
    }

    // Set the arguments to our compute kernel
//
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, _kernel->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(double) * results_count, results_array, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    // Release objects.
    clReleaseMemObject(input);
    clReleaseMemObject(output);

    // Retrieve and return first single result from output buffer.
    outputValue = &results_array[0];

    return EXIT_SUCCESS;
}
int exec_kernel(CLKernel _kernel, cl_double input_equity, double* data_array, unsigned int data_count, double** results_array)
{
    int i = 0;
    unsigned int count = data_count;
    //double data[DATA_SIZE];              // original data set given to device
    //double results[DATA_SIZE];           // results returned from device
    //unsigned int correct;               // number of correct results returned

    *results_array = (double*)malloc(sizeof(double) * data_count);

    cl_kernel kernel = _kernel->kernel;
    cl_context context = _kernel->context;
    cl_command_queue commands = _kernel->commands;

    int err;                            // error code returned from api calls
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    // Create the input and output arrays in device memory for our calculation

    input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        //exit(1);
        return EXIT_FAILURE;
    }

    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data_array, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        //exit(1);
        return EXIT_FAILURE;
    }

    // Set the arguments to our compute kernel
//
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_double), &input_equity);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, _kernel->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(double) * count, results_array, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        //exit(1);
        return EXIT_FAILURE;
    }

    // Validate our results
    //correct = 0;
    //for (i = 0; i < count; i++)
    //{
    //    if (results[i] == data[i] * data[i]) 
    //    {
    //        correct++;
    //    }
    //}

    // Print a brief summary detailing the results
    //
    //printf("Computed '%d/%d' correct values!\n", correct, count);

    // Release objects.
    clReleaseMemObject(input);
    clReleaseMemObject(output);

    return EXIT_SUCCESS;
}

int load_kernel(const char* kernelSource, CLKernel kernel_obj)
{
    CLKernel kernel = kernel_obj;
    int err;                            // error code returned from api calls

    // Fill our data set with random float values
    //
    //int i = 0;
    //unsigned int count = DATA_SIZE;
    //for (i = 0; i < count; i++)
    //    data[i] = rand() / (float)RAND_MAX;

    // Connect to a compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &kernel->device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context 
    //
    kernel->context = clCreateContext(0, 1, &kernel->device_id, NULL, NULL, &err);
    if (!kernel->context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    kernel->commands = clCreateCommandQueue(kernel->context, kernel->device_id, 0, &err);
    if (!kernel->commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    kernel->program = clCreateProgramWithSource(kernel->context, 1, (const char**)&kernelSource, NULL, &err);
    if (!kernel->program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(kernel->program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(kernel->program, kernel->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel->kernel = clCreateKernel(kernel->program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    return 0;
}

void cleanup_kernel(CLKernel kernel)
{
    // Shutdown and cleanup
    //
    //clReleaseMemObject(input);
    //clReleaseMemObject(output);
    clReleaseProgram(kernel->program);
    clReleaseKernel(kernel->kernel);
    clReleaseCommandQueue(kernel->commands);
    clReleaseContext(kernel->context);
}

// MQL4 DLLSample.
//---
#define MT4_EXPFUNC __declspec(dllexport)
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#pragma pack(push,1)
struct RateInfo
  {
   __int64           ctm;
   double            open;
   double            low;
   double            high;
   double            close;
   unsigned __int64  vol_tick;
   int               spread;
   unsigned __int64  vol_real;
  };
#pragma pack(pop)
//---
struct MqlStr
  {
   int               len;
   char             *string;
  };
static int CompareMqlStr(const void *left,const void *right);
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
BOOL APIENTRY DllMain(HANDLE hModule,DWORD ul_reason_for_call,LPVOID lpReserved)
  {
//---
   switch(ul_reason_for_call)
     {
      case DLL_PROCESS_ATTACH:
      case DLL_THREAD_ATTACH:
      case DLL_THREAD_DETACH:
      case DLL_PROCESS_DETACH:
         break;
     }
//---
   return(TRUE);
  }

MT4_EXPFUNC void _stdcall NeuralInit()
{
    _kernel_neural_forwardpass = new _CLKernel();
    load_kernel(KernelSource_NeuralForwardPass, _kernel_neural_forwardpass);

    _kernel_neural_propoutputs = new _CLKernel();   
    load_kernel(KernelSource_NeuralPropOutputs, _kernel_neural_propoutputs);
    
    _kernel_neural_training_backprop = new _CLKernel();
    load_kernel(KernelSource_NeuralTrainBackprop, _kernel_neural_training_backprop);
}

MT4_EXPFUNC void _stdcall NeuralDeinit()
{
    cleanup_kernel(_kernel_neural_forwardpass);
    delete _kernel_neural_forwardpass;

    cleanup_kernel(_kernel_neural_propoutputs);
    delete _kernel_neural_propoutputs;
}

MT4_EXPFUNC void _stdcall NeuralForwardPass(double inputEquity, double* weights, double* hiddenValues, const int arraysize)
{
    cl_double input_equity = inputEquity;
    exec_kernel(_kernel_neural_forwardpass, input_equity, weights, arraysize, &hiddenValues);

    /*
    for (int i = 0; i < NNET_hidden_count; i++)
    {
        double sum = weights[i * 0] * inputEquity + weights[i * 1];
        hiddenValues[i] = sigmoid(sum);
    }
    // */
}

MT4_EXPFUNC void _stdcall NeuralPropagateOutputs(double* outputValue, double* hiddenValues, const int arraysize)
{
    exec_kernel_singleoutput(_kernel_neural_propoutputs, outputValue, hiddenValues, arraysize);
    /*
   double outputValue = 0;
   for (int i = 0; i < NNET_hidden_count; i++)
   {
     outputValue += hiddenValues[i];
   }
   // */
}

// NeuralTrainBackwardPropagate(error, NNET_learningRate, NNET_momentum, weights, NNET_hidden_count * NNET_inputs_count);
MT4_EXPFUNC bool _stdcall NeuralTrainBackwardPropagate(double error, double NNET_learningRate, double NNET_momentum, int NNET_outputs_count,
    double* input_weights, double* output_weights, double* hiddenValues, double* deltaWeights, const int arraysize)
{
    int result = exec_kernel_training(_kernel_neural_training_backprop, error, NNET_learningRate, NNET_momentum, 
        input_weights, &output_weights, 
        hiddenValues, NNET_outputs_count, 
        &deltaWeights, arraysize);

    if (result == 0)
    {
        return TRUE;
    }
    return FALSE;

    /*
    // Backward pass
    for (int i = 0; i < NNET_hidden_count; i++)
    {
      for (int j = 0; j < NNET_outputs_count; j++)
      {
         //double delta = errors[j] * hiddenValues[i] * (1 - hiddenValues[i]);
         //deltaWeights[i][j] = NNET_learningRate * delta + NNET_momentum * deltaWeights[i][j];
         //weights[i][j] += deltaWeights[i][j];

         double delta = error * hiddenValues[i] * (1 - hiddenValues[i]);
         deltaWeights[i*j] = NNET_learningRate * delta + NNET_momentum * deltaWeights[i*j];
         weights[i*j] += deltaWeights[i*j];

      }
    }
     // */
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MT4_EXPFUNC int __stdcall GetIntValue(const int ipar)
  {
   printf("GetIntValue takes %d\n",ipar);
   return(ipar);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MT4_EXPFUNC double __stdcall GetDoubleValue(const double dpar)
  {
   printf("GetDoubleValue takes %.8lf\n",dpar);
   return(dpar);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MT4_EXPFUNC wchar_t* __stdcall GetStringValue(wchar_t *spar)
  {
   wprintf(L"GetStringValue takes \"%s\"\n",spar);
   return(spar);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MT4_EXPFUNC double __stdcall GetArrayItemValue(const double *arr,const int arraysize,const int nitem)
  {
//---
   if(arr==NULL)
     {
      printf("GetArrayItemValue: NULL array\n");
      return(0.0);
     }
   if(arraysize<=0)
     {
      printf("GetArrayItemValue: wrong arraysize (%d)\n", arraysize);
      return(0.0);
     }
   if(nitem<0 || nitem>=arraysize)
     {
      printf("GetArrayItemValue: wrong item number (%d)\n", nitem);
      return(0.0);
     }
//---
   return(arr[nitem]);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MT4_EXPFUNC bool _stdcall SetArrayItemValue(double *arr,const int arraysize,const int nitem,const double value)
  {
//---
   if(arr==NULL)
     {
      printf("GetArrayItemValue: NULL array\n");
      return(FALSE);
     }
   if(arraysize<=0)
     {
      printf("GetArrayItemValue: wrong arraysize (%d)\n", arraysize);
      return(FALSE);
     }
   if(nitem<0 || nitem>=arraysize)
     {
      printf("GetArrayItemValue: wrong item number (%d)\n", nitem);
      return(FALSE);
     }
//---
   arr[nitem]=value;
   return(TRUE);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MT4_EXPFUNC double __stdcall GetRatesItemValue(const RateInfo* rates,const int rates_total,const int shift,const int nrate)
  {
//---
   if(rates==NULL)
     {
      printf("GetRatesItemValue: NULL array\n");
      return(0.0);
     }
//---
   if(rates_total<0)
     {
      printf("GetRatesItemValue: wrong rates_total number (%d)\n", rates_total);
      return(0.0);
     }
//---
   if(shift<0 || shift>=rates_total)
     {
      printf("GetRatesItemValue: wrong shift number (%d)\n", shift);
      return(0.0);
     }
//---
   if(nrate<0 || nrate>5)
     {
      printf("GetRatesItemValue: wrong rate index (%d)\n", nrate);
      return(0.0);
     }
//---
   int nitem=rates_total-1-shift;
   switch(nrate)
     {
      case 0: return double(rates[nitem].ctm);
      case 1: return rates[nitem].open;
      case 2: return rates[nitem].low;
      case 3: return rates[nitem].high;
      case 4: return rates[nitem].close;
      case 5: return double(rates[nitem].vol_tick);
     }
//---
   return(0.0);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MT4_EXPFUNC int __stdcall SortStringArray(MqlStr *arr,const int arraysize)
  {
//---
   if(arr==NULL)
     {
      printf("SortStringArray: NULL array\n");
      return(-1);
     }
   if(arraysize<=0)
     {
      printf("SortStringArray: wrong arraysize (%d)\n", arraysize);
      return(-1);
     }
//---
   qsort(arr,arraysize,sizeof(MqlStr),CompareMqlStr);
//---
   return(arraysize);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
MT4_EXPFUNC int __stdcall ProcessStringArray(MqlStr *arr,const int arraysize)
  {
   int   len1,len2;
//---
   if(arr==NULL)
     {
      printf("ProcessStringArray: NULL array\n");
      return(-1);
     }
   if(arraysize<=0)
     {
      printf("ProcessStringArray: wrong arraysize (%d)\n", arraysize);
      return(-1);
     }
//---
   for(int i=0; i<arraysize-1; i++)
     {
      if(arr[i].string==NULL)
         len1=0;
      else
         len1=strlen(arr[i].string);

      if(arr[i+1].string==NULL)
         len2=0;
      else
         len2=strlen(arr[i+1].string);
      //--- uninitialized string
      if(arr[i+1].string==NULL)
         continue;
      //--- destination string is uninitialized and cannot be allocated within dll
      if(arr[i].string==NULL)
         continue;
      //--- memory piece is less than needed and cannot be reallocated within dll
      if(arr[i].len<len1+len2)
         continue;
      //--- final processing
      strcat(arr[i].string,arr[i+1].string);
     }
//---
   return(arraysize);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CompareMqlStr(const void *left,const void *right)
  {
   MqlStr *leftstr=(MqlStr *)left;
   MqlStr *rightstr=(MqlStr *)right;
//---
   if(leftstr->string==NULL) 
      return(-1);
   if(rightstr->string==NULL) 
      return(1);
//---
   return(strcmp(leftstr->string,rightstr->string));
  }
//+------------------------------------------------------------------+
