# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-src")
  file(MAKE_DIRECTORY "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-src")
endif()
file(MAKE_DIRECTORY
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-build"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-subbuild/triton_kernels-populate-prefix"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-subbuild/triton_kernels-populate-prefix/tmp"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-subbuild/triton_kernels-populate-prefix/src/triton_kernels-populate-stamp"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-subbuild/triton_kernels-populate-prefix/src"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-subbuild/triton_kernels-populate-prefix/src/triton_kernels-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-subbuild/triton_kernels-populate-prefix/src/triton_kernels-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/triton_kernels-subbuild/triton_kernels-populate-prefix/src/triton_kernels-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
