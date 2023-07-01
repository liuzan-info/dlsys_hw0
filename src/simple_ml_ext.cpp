#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <iostream>

namespace py = pybind11;


float* matrix_multiply(const float *a, const float *b, 
                        size_t num_a_rows, size_t num_a_cols, size_t num_b_rows, size_t num_b_cols)
{
    assert(num_a_cols == num_b_rows);
    float *result = new float[num_a_rows * num_b_cols];
    for (size_t i = 0; i < num_a_rows; i++) {
        for (size_t k = 0; k < num_b_cols; k++) {
            for (size_t j = 0; j < num_a_cols; j++) {
                result[i * num_b_cols + k] += a[i * num_a_cols + j] * b[j * num_b_cols + k];
            }
        }
    }
    return result;
}

float* softmax(const float *matrix, size_t n_rows, size_t n_cols)
{
    float *result = new float[n_rows * n_cols];
    for (size_t i = 0; i < n_rows; i++) {
        float cur_exp_sum = 0.;
        for (size_t j = 0; j < n_cols; j++) {
            result[i * n_cols + j] = exp(matrix[i * n_cols + j]);
            cur_exp_sum += result[i * n_cols + j];
        }
        for (size_t j = 0; j < n_cols; j++) {
            result[i * n_cols + j] /= cur_exp_sum;
        }
    }
    return result;
}

float* build_one_hot_matrix(const unsigned char *vec, size_t n_rows, size_t n_labels)
{
    float *result = new float[n_rows * n_labels]();
    for (size_t i = 0; i < n_rows; i++) {
        result[i * n_labels + vec[i]] = 1.0;
    }
    return result;
}

float* transpose(const float *matrix, size_t n_rows, size_t n_cols)
{
    float *result = new float[n_rows * n_cols];
    for (size_t i = 0; i < n_rows; i++) {
        for (size_t j = 0; j < n_cols; j++) {
            result[j * n_rows + i] = matrix[i * n_cols + j];
        }
    }
    return result;
}

float* matrix_subtract(const float *a, const float *b, size_t n)
{
    float *result = new float[n];
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

float *matrix_scalar_multiply(const float *a, const float x, size_t n)
{
    float *result = new float[n];
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] * x;
    }
    return result;
}

void matrix_subtract_inplace(float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        a[i] -= b[i];
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t start_index = 0;
    while (start_index < m) {
        size_t batch_size;

        // Determine batch size
        if (start_index + batch <= m){
            batch_size = batch;
        } else {
            batch_size = m - start_index;
        }

        //Initialize mini batch
        size_t num_x_elems = n * batch_size;
        float *x_mini_batch = new float[num_x_elems];
        unsigned char *y_mini_batch = new unsigned char[batch_size];
        memcpy(x_mini_batch, X + start_index * n, num_x_elems * sizeof(float));
        memcpy(y_mini_batch, y + start_index, batch_size * sizeof(unsigned char));

        float *h = matrix_multiply(x_mini_batch, theta, batch_size, n, n, k);
        float *z = softmax(h, batch_size, k);
        float *i_y = build_one_hot_matrix(y_mini_batch, batch_size, k);
        float *grad = matrix_multiply(
            transpose(x_mini_batch, batch_size, n), matrix_subtract(z, i_y, batch_size * k),
            n, batch_size, batch_size, k
        );
        float *delta = matrix_scalar_multiply(grad, lr / batch_size, n * k);
        matrix_subtract_inplace(theta, delta, n * k);
        
        start_index += batch_size;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <iostream>

namespace py = pybind11;


float* matrix_multiply(const float *a, const float *b, 
                        size_t num_a_rows, size_t num_a_cols, size_t num_b_rows, size_t num_b_cols)
{
    assert(num_a_cols == num_b_rows);
    float *result = new float[num_a_rows * num_b_cols];
    for (size_t i = 0; i < num_a_rows; i++) {
        for (size_t k = 0; k < num_b_cols; k++) {
            for (size_t j = 0; j < num_a_cols; j++) {
                result[i * num_b_cols + k] += a[i * num_a_cols + j] * b[j * num_b_cols + k];
            }
        }
    }
    return result;
}

float* softmax(const float *matrix, size_t n_rows, size_t n_cols)
{
    float *result = new float[n_rows * n_cols];
    for (size_t i = 0; i < n_rows; i++) {
        float cur_exp_sum = 0.;
        for (size_t j = 0; j < n_cols; j++) {
            result[i * n_cols + j] = exp(matrix[i * n_cols + j]);
            cur_exp_sum += result[i * n_cols + j];
        }
        for (size_t j = 0; j < n_cols; j++) {
            result[i * n_cols + j] /= cur_exp_sum;
        }
    }
    return result;
}

float* build_one_hot_matrix(const unsigned char *vec, size_t n_rows, size_t n_labels)
{
    float *result = new float[n_rows * n_labels]();
    for (size_t i = 0; i < n_rows; i++) {
        result[i * n_labels + vec[i]] = 1.0;
    }
    return result;
}

float* transpose(const float *matrix, size_t n_rows, size_t n_cols)
{
    float *result = new float[n_rows * n_cols];
    for (size_t i = 0; i < n_rows; i++) {
        for (size_t j = 0; j < n_cols; j++) {
            result[j * n_rows + i] = matrix[i * n_cols + j];
        }
    }
    return result;
}

float* matrix_subtract(const float *a, const float *b, size_t n)
{
    float *result = new float[n];
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

float *matrix_scalar_multiply(const float *a, const float x, size_t n)
{
    float *result = new float[n];
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] * x;
    }
    return result;
}

void matrix_subtract_inplace(float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        a[i] -= b[i];
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t start_index = 0;
    while (start_index < m) {
        size_t batch_size;

        // Determine batch size
        if (start_index + batch <= m){
            batch_size = batch;
        } else {
            batch_size = m - start_index;
        }

        //Initialize mini batch
        size_t num_x_elems = n * batch_size;
        float *x_mini_batch = new float[num_x_elems];
        unsigned char *y_mini_batch = new unsigned char[batch_size];
        memcpy(x_mini_batch, X + start_index * n, num_x_elems * sizeof(float));
        memcpy(y_mini_batch, y + start_index, batch_size * sizeof(unsigned char));

        float *h = matrix_multiply(x_mini_batch, theta, batch_size, n, n, k);
        float *z = softmax(h, batch_size, k);
        float *i_y = build_one_hot_matrix(y_mini_batch, batch_size, k);
        float *grad = matrix_multiply(
            transpose(x_mini_batch, batch_size, n), matrix_subtract(z, i_y, batch_size * k),
            n, batch_size, batch_size, k
        );
        float *delta = matrix_scalar_multiply(grad, lr / batch_size, n * k);
        matrix_subtract_inplace(theta, delta, n * k);
        
        start_index += batch_size;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
