// Tiny compute-capability probe, built with -arch=native before the real build.
// Prints machine-readable lines that RUN_ME.bat parses:
//   SMMAJOR=<n> / SMMINOR=<n> / DEVNAME=<name>
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        printf("SMMAJOR=0\nSMMINOR=0\nDEVNAME=NONE\n");
        return 2;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("SMMAJOR=%d\nSMMINOR=%d\nDEVNAME=%s\n", prop.major, prop.minor, prop.name);
    return 0;
}
