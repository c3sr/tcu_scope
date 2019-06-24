
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)


option(CONFIG_USE_TRAVIS "Turn on to enable using travis mode" OFF)


if (UNIX AND NOT CONFIG_USE_TRAVIS)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  set(CUDACXX /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_HOST_COMPILER gcc)
endif(UNIX AND NOT CONFIG_USE_TRAVIS)

if (APPLE)
  set(CMAKE_C_COMPILER gcc-5)
  set(CMAKE_CXX_COMPILER g++-5)
  set(CUDACXX /Developer/NVIDIA/CUDA-9.2/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /Developer/NVIDIA/CUDA-9.2/bin/nvcc)
  set(CMAKE_CUDA_COMPILER_FORCED ON)
  set(CMAKE_CUDA_HOST_COMPILER g++-5)
endif(APPLE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include("cmake/modules/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.20.41.tar.gz"
    SHA1 "e8e77b69fc2ca66808091e781cc06bcb02724443"
)



project(tensorcore_bench LANGUAGES CXX CUDA VERSION 0.1.0)


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

include(GetGitRevisionDescription)
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)





hunter_add_package(sugar)
find_package(sugar CONFIG REQUIRED)

hunter_add_package(benchmark)
find_package(benchmark CONFIG REQUIRED)

hunter_add_package(nlohmann_json)
find_package(nlohmann_json CONFIG REQUIRED)

hunter_add_package(fmt)
find_package(fmt CONFIG REQUIRED)

hunter_add_package(spdlog)
find_package(spdlog CONFIG REQUIRED)

hunter_add_package(cub)
find_package(cub CONFIG REQUIRED)

hunter_add_package(debug_assert)
find_package(debug_assert CONFIG REQUIRED)


find_package(CUDA REQUIRED)
find_package(OpenBLAS REQUIRED)


include(sugar_doxygen_generate)
include(sugar_include)

# Disable extended variants of C++ dialects
# i.e. don't choose gnu++17 over c++17
set(CMAKE_CXX_EXTENSIONS OFF)


if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wundef -fcolor-diagnostics -stdlib=libc++ -pthread "
  CACHE STRING "Toolchain C++ compiler flags." FORCE)


set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++ -lc++abi -fuse-ld=lld"
  CACHE STRING "Toolchain C++ linker flags." FORCE)
endif()

if (UNIX AND NOT APPLE)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER}
                  -fuse-ld=gold -Wl,--version
                  ERROR_QUIET OUTPUT_VARIABLE ld_version)
  if ("${ld_version}" MATCHES "GNU gold")
    message(STATUS "Found Gold linker, use faster linker")
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold ")
  endif()
endif()



if(MSVC)
  # /wd4005  macro-redefinition
  # /wd4068  unknown pragma
  # /wd4244  conversion from 'type1' to 'type2'
  # /wd4267  conversion from 'size_t' to 'type2'
  # /wd4800  force value to bool 'true' or 'false' (performance warning)
  add_compile_options(/wd4005 /wd4068 /wd4244 /wd4267 /wd4800)
  add_definitions(/DNOMINMAX /DWIN32_LEAN_AND_MEAN=1 /DCONSTEXPR=const /D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DCONSTEXPR=constexpr)
	set(CXX_WARNINGS "-Wall -Wextra -Wno-unused-function -Wno-unknown-warning-option -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs -Wno-misleading-indentation -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough -Wno-unused-variable -Wno-unused-command-line-argument")
  list(APPEND CMAKE_CXX_FLAGS "-fmax-errors=1 -Wfatal-errors ${CXX_WARNINGS} ")
endif()


if (APPLE)
  list (APPEND CMAKE_CXX_FLAGS_INIT "-fvisibility=hidden -fvisibility-inlines-hidden")
endif(APPLE)

# CUDA flags
set(CMAKE_CUDA_STANDARD 11)


CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Volta")



#list(APPEND CMAKE_CXX_FLAGS "-g -ggdb -DDEBUG=1")

LIST(APPEND CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS}
                   -Xcompiler -std=c++11
                   -Xcompiler -Wno-vla
                   -Xcompiler -rdynamic
				   -Wno-deprecated-gpu-targets
                   --expt-extended-lambda
                   --expt-relaxed-constexpr
                   -arch=sm_70
                   # -rdc=true # enable if you want to use the cg benchmarks
                   -use_fast_math
                   #-res-usage
                   #-keep
                   #-keep-dir
                   #-lineinfo
                   #-g -pg
)
string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


MESSAGE(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
			   " archs: ${CUDA_ARCH_FLAGS_readable})")

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)



set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(THIRDPARTY_DIR "${TOP_DIR}/third_party")
set(SRC_DIR "${TOP_DIR}/src")

sugar_include("./src")


add_executable(bench ${BENCHMARK_SOURCES} ${BENCHMARK_CUDA_SOURCES})


include_directories(bench
  ${SRC_DIR}
  ${THIRDPARTY_DIR}
  ${THIRDPARTY_DIR}/cutlass
  ${TOP_DIR}/include
  ${OpenBLAS_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

add_definitions(-DPROJECT_NAME=tensorcore_bench)

set_target_properties(bench PROPERTIES
    #CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(bench benchmark::benchmark)
target_link_libraries(bench nlohmann_json)
target_link_libraries(bench fmt::fmt)
target_link_libraries(bench spdlog::spdlog)
target_link_libraries(bench cub::cub)
target_link_libraries(bench debug_assert)
target_link_libraries(bench ${OpenBLAS_LIB})
target_link_libraries(bench ${CUDA_LIBRARIES})
target_link_libraries(bench ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(bench ${CUDA_curand_LIBRARY})


# Request that bench be built with -std=c++11
# As this is a public compile feature anything that links to
# bench will also build with -std=c++11
target_compile_features(bench PUBLIC cxx_std_11)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET bench
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()
