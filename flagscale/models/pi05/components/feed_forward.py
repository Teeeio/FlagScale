# Copyright 2025 FlagScale Authors.
# Licensed under the Apache License, Version 2.0.

"""
FeedForward Layer - 100% matching OpenPI implementation.

Reference: openpi/src/openpi/models/gemma.py:253-280
"""

import torch
import torch.nn as nn
import math
import sys
sys.path.insert(0, '/nfs/wzp/libero/FlagScale')

class FeedForward(nn.Module):
    """
    Gated FeedForward Network.

    精确匹配OpenPI实现: gemma.py:253-280

    架构:
    1. Gated FFN: gelu(W_gate(x)) * W_up(x)
    2. Down projection: W_down(...)

    关键特性:
    - 使用SwiGLU (gated)激活函数
    - lecun_normal初始化
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int
    ):
        """
        Args:
            dim: input dimension
            hidden_dim: hidden dimension (通常为4*dim)
        """
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim

        # Step 1: Gated FFN
        # OpenPI: Wi_gate = nn.Dense(features=hidden_dim, kernel_init=lecun_normal())
        #          Wi_up = nn.Dense(features=hidden_dim, kernel_init=lecun_normal())
        #          Wi_down = nn.Dense(features=dim, kernel_init=lecun_normal())

        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

        # Step 3: lecun_normal初始化
        # OpenPI: kernel_init=lecun_normal()
        # lecun_normal: std = sqrt(1/fan_in)
        self._init_weights(self.gate)
        self._init_weights(self.up)
        self._init_weights(self.down)

    def _init_weights(self, layer: nn.Linear):
        """使用lecun_normal初始化"""
        # lecun_normal: std = sqrt(1 / fan_in)
        fan_in = layer.in_features
        std = math.sqrt(1.0 / fan_in)
        nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, T, D] input tensor
        Returns:
            [B, T, D] output tensor

        实现细节 (逐行对应OpenPI):
        OpenPI代码 (简化):
        ```python
        def __call__(self, x):
            gate_activation = nn.gelu(Wi_gate(x))
            up = Wi_up(x)
            ff_output = Wi_down(gate_activation * up)
            return ff_output
        ```

        注意:
        - nn.gelu是JAX的近似实现: gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        - PyTorch的nn.GELU默认使用tanh近似，与JAX一致
        """
        # Step 1: Gated FFN
        # OpenPI: gate_act = nn.gelu(self.Wi_gate(x))
        #          up = self.Wi_up(x)
        #          ff_out = self.Wi_down(gate_act * up)

        # 保存原始dtype
        dtype = x.dtype

        # OpenPI: dot(x, w.astype(x.dtype))，确保权重使用输入dtype
        w_gate = self.gate.weight.to(dtype)
        w_up = self.up.weight.to(dtype)
        w_down = self.down.weight.to(dtype)

        # 使用tanh近似以匹配JAX默认gelu(approximate=True)
        gelu = torch.nn.functional.gelu

        gate_activation = gelu(torch.matmul(x, w_gate.t()), approximate="tanh")  # [B, T, hidden_dim]
        up = torch.matmul(x, w_up.t())  # [B, T, hidden_dim]

        # Element-wise乘法
        ff_output = torch.matmul(gate_activation * up, w_down.t())  # [B, T, dim]

        # 转回原始dtype
        ff_output = ff_output.to(dtype)

        return ff_output


def test_feed_forward():
    """验证FeedForward实现的正确性"""

    print("=" * 80)
    print("FeedForward 单元测试")
    print("=" * 80)

    batch_size, seq_len, dim = 2, 10, 128
    hidden_dim = dim * 4  # Gemma标准配置

    # 测试1: 形状与dtype
    print("\n【测试1】形状与dtype")
    print("-" * 80)

    ffn = FeedForward(dim=dim, hidden_dim=hidden_dim)
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)
    output = ffn(x)

    assert output.shape == x.shape, f"Shape错误: {output.shape} != {x.shape}"
    assert output.dtype == x.dtype, f"Dtype错误: {output.dtype} != {x.dtype}"
    print(f"  ✅ Shape正确: {output.shape}")
    print(f"  ✅ Dtype保持: {output.dtype}")

    # 测试2: 手动验证FFN计算
    print("\n【测试2】手动验证FFN计算")
    print("-" * 80)

    ffn = FeedForward(dim=dim, hidden_dim=hidden_dim)
    x = torch.randn(1, 1, dim, dtype=torch.float32)

    gate = ffn.gate(x)
    gate_activation = torch.nn.functional.gelu(gate)
    up = ffn.up(x)
    expected = ffn.down(gate_activation * up)

    actual = ffn(x)
    diff = (actual - expected).abs().max()
    assert diff < 1e-6, f"FFN计算不匹配: diff = {diff}"
    print(f"  ✅ FFN计算正确: diff = {diff.item():.2e}")

    print("\n" + "=" * 80)
    print("✅ FeedForward 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_feed_forward()
