# HierMoE NPU Planner Operators

These optional Ascend C operators accelerate the exact current-route HierMoE
planner. The Python planner loads them automatically when the package-local
extension has been built; CPU, GPU, and unbuilt NPU environments retain the
exact PyTorch fallback.

The package contains three exact kernels: route packing, replica candidate
scoring, and accepted-candidate state application. The state application uses
one AI vector core because its scalar updates touch neighboring cache lines;
candidate scoring remains parallel across all available cores. Its mutable
state is registered explicitly with the PyTorch dispatcher.

Build inside an Ascend development environment:

```bash
bash veomni/ops/platform/npu/hiermoe_planner_ops/build.sh
```

The generated OPP package and Python extension remain under this directory and
are ignored rather than committed. Source distributions and wheels include the
operator sources so NPU installations can build the optional extension in
place.

`csrc/extension/pytorch_npu_helper.hpp` is derived from Huawei Ascend operator
samples and retains its upstream BSD 3-Clause license header. The remaining
sources are licensed under the repository's Apache License 2.0 terms.
