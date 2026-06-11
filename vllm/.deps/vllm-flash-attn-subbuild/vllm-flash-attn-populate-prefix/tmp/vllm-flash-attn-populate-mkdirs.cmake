# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/vllm-flash-attn-src")
  file(MAKE_DIRECTORY "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/vllm-flash-attn-src")
endif()
file(MAKE_DIRECTORY
  "/tmp/tmpzqcm6v58.build-temp/vllm-flash-attn"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/tmp"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src/vllm-flash-attn-populate-stamp"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src"
  "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src/vllm-flash-attn-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src/vllm-flash-attn-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ubuntu/ali_vllm/vllm_dflash/vllm/.deps/vllm-flash-attn-subbuild/vllm-flash-attn-populate-prefix/src/vllm-flash-attn-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
