#ifndef __GQD_CU__
#define __GQD_CU__

/**
 * the API file
 * includes every thing for this library
 */

/* gdd_library */
#include "gdd_basic.cu"
#include "gdd_sqrt.cu"
#include "gdd_exp.cu"
#include "gdd_log.cu"
#include "gdd_sincos.cu"

#if 0
/* gqd_libraray */
#include "gqd_basic.cu"
#include "gqd_sqrt.cu"
#include "gqd_exp.cu"
#include "gqd_log.cu"
#include "gqd_sincos.cu"

#endif

// high to low precision
#define copy_vec_element(b, a, c) b.c = double((a).c);

#define init_vec2(b, a) copy_vec_element(b, a, x) copy_vec_element(b, a, y)
#define init_vec3(b, a) copy_vec_element(b, a, x) copy_vec_element(b, a, y) copy_vec_element(b, a, z)
#define init_vec4(b, a) copy_vec_element(b, a, x) copy_vec_element(b, a, y) copy_vec_element(b, a, z) copy_vec_element(b, a, w)

#define define_vec2(type, b, a) type b; init_vec2(b, a)
#define define_vec3(type, b, a) type b; init_vec3(b, a)
#define define_vec4(type, b, a) type b; init_vec4(b, a)

#define make_vec2(type, a) make_##type##2(a.x, a.y)
#define make_vec3(type, a) make_##type##3(a.x, a.y, a.z)
#define make_vec4(type, a) make_##type##4(a.x, a.y, a.z, a.w)

template <class Src, class Dst >
__host__ __device__ Dst ValueConverter(Src dv) {
    Dst b = Dst(dv); return b;
}

template <class Src, class Dst >
__host__ __device__ Dst ValueConverter2(Src dv) {
    define_vec2(Dst, b, dv); return b;
}

template <class Src, class Dst >
__host__ __device__ Dst ValueConverter3(Src dv) {
    define_vec3(Dst, b, dv); return b;
}

template <class Src, class Dst >
__host__ __device__ Dst ValueConverter4(Src dv) {
    define_vec4(Dst, b, dv); return b;
}

template <class Dst, class Src, unsigned int size >
class ArrayConverterHelper {
public:
    __host__ __device__ ArrayConverterHelper(Src srcArray[]) {
        this->srcArray = (Src*)srcArray;
        for (unsigned int i = 0; i < size; i++) {
            dstArray[i] = Dst(srcArray[i]);
        }
    }
    __host__ __device__ ~ArrayConverterHelper() {
        for (unsigned int i = 0; i < size; i++) {
            srcArray[i] = Src(dstArray[i]);
        }        
    }
    Dst dstArray[size];
    Src* srcArray;
    __host__ __device__ Dst* getArray() { return dstArray; }

    __host__ __device__ static void Convert(Dst* dst, Src* src) {
        for (unsigned int i = 0; i < size; i++) {
            dst[i] = Dst(src[i]);
        }
    }
};

template <class Src, class Dst >
class RefConverter {
public:
    __host__ __device__ RefConverter(Src& dv) { 
        this->dv = &dv;
        this->v = Dst(dv);
    }
    __host__ __device__ RefConverter(Src* pdv) {
        this->dv = pdv;
        this->v = Dst(*pdv);
    }
    __host__ __device__ ~RefConverter() { 
        *dv = Src(v);
    }
    Dst v;
    Src* dv;
    __host__ __device__ Dst& ref() { return v; }
    __host__ __device__ Dst* ptr() { return &v; }
};

template <class Src, class Dst >
class RefConverter2 {
public:
    __host__ __device__ RefConverter2(Src& dv) { 
        this->dv = &dv;
        init_vec2(v, dv);
    }
    __host__ __device__ RefConverter2(Src* pdv) { 
        this->dv = pdv;
        init_vec2(v, (*pdv));
    }
    __host__ __device__ ~RefConverter2() { 
        init_vec2((*dv), v);
    }
    Dst v;
    Src* dv;
    __host__ __device__ Dst& ref() { return v; }
    __host__ __device__ Dst* ptr() { return &v; }
};

template <class Src, class Dst >
class RefConverter3 {
public:
    __host__ __device__ RefConverter3(Src& dv) { 
        this->dv = &dv;
        init_vec3(v, dv);
    }
    __host__ __device__ RefConverter3(Src* pdv) { 
        this->dv = pdv;
        init_vec3(v, (*pdv));
    }
    __host__ __device__ ~RefConverter3() { 
        init_vec3((*dv), v);
    }
    Dst v;
    Src* dv;
    __host__ __device__ Dst& ref() { return v; }
    __host__ __device__ Dst* ptr() { return &v; }
};

template <class Src, class Dst >
class RefConverter4 {
public:
    __host__ __device__ RefConverter4(Src& dv) { 
        this->dv = &dv;
        init_vec4(v, dv);
    }
    __host__ __device__ RefConverter4(Src* pdv) { 
        this->dv = pdv;
        init_vec4(v, (*pdv));
    }
    __host__ __device__ ~RefConverter4() { 
        init_vec4((*dv), v);
    }
    Dst v;
    Src* dv;
    __host__ __device__ Dst& ref() { return v; }
    __host__ __device__ Dst* ptr() { return &v; }
};

#define ConvertArray(dst, src, a) ArrayConverterHelper<dst, src, sizeof(a)/sizeof(a[0])>(a).getArray()
#define CopyArray(dst, src, b, a) ArrayConverterHelper<dst, src, sizeof(a)/sizeof(a[0])>::Convert(b, a)

#define ConvertRef(dst, src, a) RefConverter<src, dst>(a).ref()
#define ConvertRef2(dst, src, a) RefConverter2<src, dst>(a).ref()
#define ConvertRef3(dst, src, a) RefConverter3<src, dst>(a).ref()
#define ConvertRef4(dst, src, a) RefConverter4<src, dst>(a).ref()

#define ConvertValue(dst, src, a) ValueConverter<src, dst>(a)
#define ConvertValue2(dst, src, a) ValueConverter2<src, dst>(a)
#define ConvertValue3(dst, src, a) ValueConverter3<src, dst>(a)
#define ConvertValue4(dst, src, a) ValueConverter4<src, dst>(a)

class GQDInitObj {
public:
    GQDInitObj() { GDDStart(); }
    ~GQDInitObj() { GDDEnd(); }
};

static __device__ double gdd_to_double(gdd_real a) { return a.x; }
static __device__ gdd_real double_to_gdd(double a) { return make_dd(a); }
static __device__ gdd_real double_to_gdd(gdd_real a) { return a; }

typedef long double float_80;

inline void cudaErrorCheck(cudaError_t code) {
    if (code != cudaSuccess) {
        printf("kernel error: %s\n", cudaGetErrorString(code));
    }
}


#endif // __GQD_CU__


