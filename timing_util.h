#ifndef GPU_ROWHAMMER_TIMING_UTIL_H
#define GPU_ROWHAMMER_TIMING_UTIL_H
long long toNS(long long time) { return (time / 1545000000.0f) * 1000000000.0; }

#endif /* GPU_ROWHAMMER_TIMING_UTIL_H */