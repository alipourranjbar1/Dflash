// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Torch library registration for FlashMLA

#include <Python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "pytorch_shim.h"
#include "api/common.h"
#include "api/dense_decode.h"
#include "api/dense_fwd.h"
#include "api/sparse_decode.h"
#include "api/sparse_fwd.h"


TORCH_LIBRARY(_flashmla_C, m) {
    m.def("sparse_decode_fwd", make_pytorch_shim(&sparse_attn_decode_interface));
    m.impl("sparse_decode_fwd", torch::kCUDA,
           make_pytorch_shim(&sparse_attn_decode_interface));

    m.def("dense_decode_fwd", make_pytorch_shim(&dense_attn_decode_interface));
    m.impl("dense_decode_fwd", torch::kCUDA,
           make_pytorch_shim(&dense_attn_decode_interface));

    m.def("sparse_prefill_fwd", make_pytorch_shim(&sparse_attn_prefill_interface));
    m.impl("sparse_prefill_fwd", torch::kCUDA,
           make_pytorch_shim(&sparse_attn_prefill_interface));

    m.def("dense_prefill_fwd", make_pytorch_shim(&FMHACutlassSM100FwdRun));
    m.impl("dense_prefill_fwd", torch::kCUDA,
           make_pytorch_shim(&FMHACutlassSM100FwdRun));
}

PyMODINIT_FUNC PyInit__flashmla_C() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_flashmla_C", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
