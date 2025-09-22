#include "utils.hpp"
#include <cstddef>

piState piGemm(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k, size_t n)
{
    for (size_t i = 0; i < m; i++)
    {

        for (size_t j = 0; j < n; j++)
        {
            size_t offset_c = i * n + j;

            double sum = 0;
            for (size_t l = 0; l < k; l++)
            {
                size_t offset_a = i * k + l;
                size_t offset_b = l * n + j;
                sum += A[offset_a] * B[offset_b];
            }
            C[offset_c] = alpha * sum + beta * C[offset_c];
        }
    }

    return piSucsess;
}

piState piGeMV(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k)
{
    return piSucsess;
}