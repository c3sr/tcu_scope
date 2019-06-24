
option(OpenBLAS "Path to OpenBLAS Installation Directory" "")

SET(Open_BLAS_INCLUDE_SEARCH_PATHS
      ${OpenBLAS}
      /opt/DL/openblas/include
      /usr/include/x86_64-linux-gnu/
      /usr/include
      /usr/include/openblas
      /usr/include/openblas-base
      /usr/local/include
      /usr/local/include/openblas
      /usr/local/include/openblas-base
      /opt/OpenBLAS/include
      /usr/local/opt/openblas/include
      $ENV{OpenBLAS}
      $ENV{OpenBLAS_HOME}
      $ENV{OpenBLAS_HOME}/include
)

SET(Open_BLAS_LIB_SEARCH_PATHS
      ${OpenBLAS}/lib
      /opt/DL/openblas/lib
      /usr/lib/x86_64-linux-gnu/
      /lib/
      /lib/openblas-base
      /lib64/
      /usr/lib
      /usr/lib/openblas-base
      /usr/lib64
      /usr/local/lib
      /usr/local/lib64
      /opt/OpenBLAS/lib
      /usr/local/opt/openblas/lib
      $ENV{OpenBLAS}cd
      $ENV{OpenBLAS}/lib
      $ENV{OpenBLAS_LIB}
      $ENV{OpenBLAS_HOME}
      $ENV{OpenBLAS_HOME}/lib
)

FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS} NO_DEFAULT_PATH )
FIND_LIBRARY(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS} NO_DEFAULT_PATH )


SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
  SET(OpenBLAS_FOUND OFF)
  MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
  SET(OpenBLAS_FOUND OFF)
  MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
IF (NOT OpenBLAS_FIND_QUIETLY)
  MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
  MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
IF (OpenBLAS_FIND_REQUIRED)
  MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

MARK_AS_ADVANCED(
  OpenBLAS_INCLUDE_DIR
  OpenBLAS_LIB
  OpenBLAS
)
