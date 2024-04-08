#include <omp.h>
#include <iostream>
#include <cmath>
void ReLU(float *input, float *output, int n)
{
#pragma omp for
    for (int i = 0; i < n; i++)
    {
        output[i] = fmax(input[i], 0.0f);
    }
}
void Sigmoid(float *input, float *output, int n)
{
#pragma omp for
    for (int i = 0; i < n; i++)
    {
        output[i] = 1.0f / (1 + exp(input[i]));
    }
}