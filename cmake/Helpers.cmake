#
# Copyright 2017 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

include(CMakeParseArguments)


#
# create a library in the absl namespace
#
# parameters
# SOURCES : sources files for the library
# PUBLIC_LIBRARIES: targets and flags for linking phase
# PRIVATE_COMPILE_FLAGS: compile flags for the library. Will not be exported.
# EXPORT_NAME: export name for the absl:: target export
# TARGET: target name
#
# create a target associated to <NAME>
# libraries are installed under CMAKE_INSTALL_FULL_LIBDIR by default
#
function(bench_library)
  cmake_parse_arguments(BENCH_LIB
    "DISABLE_INSTALL" # keep that in case we want to support installation one day
    "TARGET;EXPORT_NAME"
    "SOURCES;PUBLIC_LIBRARIES;PRIVATE_COMPILE_FLAGS;PUBLIC_INCLUDE_DIRS;PRIVATE_INCLUDE_DIRS"
    ${ARGN}
  )

  set(_NAME ${BENCH_LIB_TARGET})
  string(TOUPPER ${_NAME} _UPPER_NAME)

  add_library(${_NAME} STATIC ${BENCH_LIB_SOURCES})

  target_compile_options(${_NAME} PRIVATE ${BENCH_COMPILE_CXXFLAGS} ${BENCH_LIB_PRIVATE_COMPILE_FLAGS})
  target_link_libraries(${_NAME} PUBLIC ${BENCH_LIB_PUBLIC_LIBRARIES})
  target_include_directories(${_NAME}
    PUBLIC ${BENCH_COMMON_INCLUDE_DIRS} ${BENCH_LIB_PUBLIC_INCLUDE_DIRS}
    PRIVATE ${BENCH_LIB_PRIVATE_INCLUDE_DIRS}
  )

  if(BENCH_LIB_EXPORT_NAME)
    add_library(absl::${BENCH_LIB_EXPORT_NAME} ALIAS ${_NAME})
  endif()
endfunction()



#
# header only virtual target creation
#
function(bench_header_library)
  cmake_parse_arguments(BENCH_HO_LIB
    "DISABLE_INSTALL"
    "EXPORT_NAME;TARGET"
    "PUBLIC_LIBRARIES;PRIVATE_COMPILE_FLAGS;PUBLIC_INCLUDE_DIRS;PRIVATE_INCLUDE_DIRS"
    ${ARGN}
  )

  set(_NAME ${BENCH_HO_LIB_TARGET})

  set(__dummy_header_only_lib_file "${CMAKE_CURRENT_BINARY_DIR}/${_NAME}_header_only_dummy.cc")

  if(NOT EXISTS ${__dummy_header_only_lib_file})
    file(WRITE ${__dummy_header_only_lib_file}
      "/* generated file for header-only cmake target */
      namespace bench {
       // single meaningless symbol
       void ${_NAME}__header_fakesym() {}
      }  // namespace bench
      "
    )
  endif()


  add_library(${_NAME} ${__dummy_header_only_lib_file})
  target_link_libraries(${_NAME} PUBLIC ${BENCH_HO_LIB_PUBLIC_LIBRARIES})
  target_include_directories(${_NAME}
    PUBLIC ${BENCH_COMMON_INCLUDE_DIRS} ${BENCH_HO_LIB_PUBLIC_INCLUDE_DIRS}
    PRIVATE ${BENCH_HO_LIB_PRIVATE_INCLUDE_DIRS}
  )

  if(BENCH_HO_LIB_EXPORT_NAME)
    add_library(absl::${BENCH_HO_LIB_EXPORT_NAME} ALIAS ${_NAME})
  endif()

endfunction()