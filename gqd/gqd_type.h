#ifndef __GDD_TYPE_H__
#define __GDD_TYPE_H__

#include <vector_types.h>


/* compiler switch */
/**
 * ALL_MATH will include advanced math functions, including
 * atan, acos, asin, sinh, cosh, tanh, asinh, acosh, atanh
 * WARNING: these functions take long time to compile, 
 * e.g., several hours
 * */
//#define ALL_MATH


/* type definition */
struct gdd_real {
    double x, y;

    __host__ __device__ gdd_real(const double a) {
        x = a; y = 0.0;
    }

    __host__ __device__ gdd_real() {
    }

    __host__ __device__ explicit operator double() {
        return x;
    }

    __host__ __device__ explicit operator int() {
        return int(x);
    }
};

typedef double4 gqd_real;


/* initialization functions, these can be called by hosts */
void GDDStart(const int device = 0);
void GDDEnd();
void GQDStart(const int device = 0);
void GQDEnd();

#endif /*__GDD_GQD_TYPE_H__*/
