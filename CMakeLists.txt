cmake_minimum_required(VERSION 3.28)

# The current implementation supports Ampere sm_86 and early Ada sm_89. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(86|89)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=86 or 89; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling

if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  add_compile_options(
    # MSVC Host C/C++
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:C,CXX>:/O2>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Ob3>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Oi>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Ot>
    $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>

    # NVCC Host Pass-through
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Ob3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Oi>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Ot>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX2>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/w14305>   # Warn on accidental double literal truncation (e.g. 1.0 vs 1.0f)
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/w14244>   # Warn on float/int precision loss

    # NVVM & PTXAS Device Optimizations
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>           # Print exact register usage and .local stack spills
    $<$<COMPILE_LANGUAGE:CUDA>:-ftz=true>            # Flush subnormals to zero (hardware fast path on sm_89)
    $<$<COMPILE_LANGUAGE:CUDA>:-prec-div=false>      # Fast single-precision hardware division (__fdividef)
    $<$<COMPILE_LANGUAGE:CUDA>:-prec-sqrt=false>     # Fuse 1.0f/sqrtf(x) into single-cycle rsqrtf(x)
  )

  # LINKER: prefix ensures CMake passes this strictly to link.exe via -Xlinker for CUDA targets
  add_link_options("LINKER:/NODEFAULTLIB:LIBCMT")
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
  add_compile_definitions(NINFER_SM86=1)
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  add_compile_definitions(NINFER_SM89=1)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "NInfer requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(WIN32)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    find_package(CURL 7.85 REQUIRED)
    set(NINFER_CURL_TARGET CURL::libcurl)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
endif()
if(NINFER_BUILD_SERVE)
  find_package(Threads REQUIRED)
  include(cmake/ninfer_ui.cmake)
endif()

if(WIN32)
  include(FetchContent)
  FetchContent_Declare(
    directstorage
    URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DirectStorage/1.3.0"
    DOWNLOAD_NAME "directstorage.1.3.0.zip"
  )
  FetchContent_MakeAvailable(directstorage)

  add_library(ninfer_directstorage INTERFACE)
  target_include_directories(ninfer_directstorage INTERFACE
    ${directstorage_SOURCE_DIR}/native/include
  )
  target_link_libraries(ninfer_directstorage INTERFACE
    ${directstorage_SOURCE_DIR}/native/lib/x64/dstorage.lib
    d3d12.lib
    dxgi.lib
  )

  set(DSTORAGE_DLLS
    "${directstorage_SOURCE_DIR}/native/bin/x64/dstorage.dll"
    "${directstorage_SOURCE_DIR}/native/bin/x64/dstoragecore.dll"
  )
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
