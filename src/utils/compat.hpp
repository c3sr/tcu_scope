#pragma once

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else // __GNUC__
#define UNUSED
#endif // __GNUC__

#if defined(__CUDA_ARCH__)
#define ALWAYS_INLINE inline
#elif defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#else // defined(__GNUC__)
#define ALWAYS_INLINE __forceinline
#endif // defined(__GNUC__)