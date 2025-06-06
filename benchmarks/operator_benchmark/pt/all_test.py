import operator_benchmark as op_bench

import torch


"""Microbenchmarks for all operators."""

cross_product_config = {
    "device": ["cpu", "cuda"],
    "dtype": [torch.float],
}



binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["sub", torch.sub],
        ["div", torch.div],
        ["mul", torch.mul],
    ],
)

binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs=cross_product_config,
)
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[128, 128], [128, 128]],
        [[256, 256], [128, 128]],
        [[512, 512], [512, 512]],
        [[1024, 1024], [1024, 1024]]
    ],
    cross_product_configs=cross_product_config,
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": torch.randn(in_one, device=device).to(dtype=dtype),
            "in_two": torch.randn(in_two, device=device).to(dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_configs_broadcast, BinaryOpBenchmark
)
op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_configs, BinaryOpBenchmark
)


# ---

unary_ops_configs_short = op_bench.config_list(
    attr_names=["in_one"],
    attrs=[
        [[128, 128]],
        [[256, 256]],
        [[512, 512]],
        [[1024, 1024]]
    ],
    cross_product_configs=cross_product_config,
)

class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, device, op_func):
        self.inputs = {"input": torch.rand(in_one, device=device)}
        self.op_func = op_func

    def forward(self, input):
        return self.op_func(input)

def clamp(input):
    return torch.clamp(input, min=0.25, max=0.75)

unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["abs", torch.abs],
        ["clamp", clamp],
        ["cos", torch.cos],
        ["exp", torch.exp],
        ["log", torch.log],
        ["neg", torch.neg],
        ["sin", torch.sin],
        ["sqrt", torch.sqrt],
        ["tan", torch.tan],
    ],
)


op_bench.generate_pt_tests_from_op_list(
    unary_ops_list, unary_ops_configs_short + unary_ops_configs_long, UnaryOpBenchmark
)




if __name__ == "__main__":
    op_bench.benchmark_runner.main()
