#ifndef CUDAUTIL_H
#define CUDAUTIL_H


inline constexpr int kCudaUtilsBufferSize = 1024;


// AMGX API error checking
#define AMGX_CHECK(err)                                                                            \
    do {                                                                                           \
        AMGX_RC err_ = (err);                                                                      \
        if (err_ != AMGX_RC_OK) {                                                                  \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "AMGX error %d at %s:%d\n", err_, __FILE__, __LINE__);         \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "cuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "CUDA API error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


// CUDA kernel function launch status checking
#define CUDA_CHECK_LAST_ERROR()                                                                    \
    do {                                                                                           \
        cudaError_t err_ = cudaGetLastError();                                                     \
        if (err_ != cudaSuccess) {                                                                 \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "CUDA kernel error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "cuSOLVER error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            char check_buf[kCudaUtilsBufferSize] {'\0'};                                           \
            std::sprintf(check_buf, "cuSPARSE error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
            throw std::runtime_error(check_buf);                                                   \
        }                                                                                          \
    } while (false)


#endif  // CUDAUTIL_H
