# 确定性反向

训练重跑时，CANN 内置的 flash attention 反向（`dq`/`dk`/`dv`）和 `npu_rms_norm`
反向（`dgamma`）的归约顺序随核数/切分变化，同一份输入两次运行会得到不同的梯度。
本仓默认用自研 Ascend C 算子替换这两处反向，使同一输入的输出逐位可复现；不依赖
`torch_npu.npu.set_deterministic_level(1)`，也不使用 torch 原生算子。

## 组成

| 算子 | 源码 | 说明 |
|---|---|---|
| `FlaRmsNormBwdDx` / `FlaRmsNormBwdDg` | `fla/ops/ascendc/det/fla_rms_norm_bwd_dx`、`.../fla_rms_norm_bwd_dg` | RMSNorm 反向的 `dx` 与 `dgamma`。`dx` 三条路径：整块 tiled（小 `d`）、整行（任意 `d`）、按列分块（大 `d`），求和顺序只取决于输入与 tiling，没有跨核归约。Python 侧由 `fla_npu.ops.det_rms_norm` 在 import 时重定向。 |
| `FlashAttentionScoreGrad` | `fla/ops/ascendc/det/flash_attention_score_grad` | flash attention 反向。移植自 ops-transformer 的 arch32 实现，确定性模板被硬编码为唯一选择。算子名与 CANN 内置算子相同，随 fla_npu 的自定义 OPP vendor 注册，靠 `ASCEND_CUSTOM_OPP_PATH` 的优先级覆盖内置算子——`torch_npu` 的 attention 反向会自动走到它，不需要任何 Python 侧改动。算子自带的三个内核辅助头（`util.h`/`pse.h`/`dropmask.h`）放在该算子目录下的 `common/`，由 `op_host/CMakeLists.txt` 的 `-I` 引入，不占用共享目录。 |

两者都随同一个 OPP 包、同一个 vendor 分发：装好 wheel 即生效，不需要额外设置
`ASCEND_CUSTOM_OPP_PATH`。

## 构建

```
FLA_NPU_SOC=<soc> python scripts/build_wheel.py
```

一个命令完成：编译 OPP（包含上述算子）→ 生成 run 包 → 组装 wheel。FA 反向的
kernel 源码量较大，首次全量编译时间明显长于只编前向算子的情况。

## 开关

默认开启。设置 `FLA_NPU_DISABLE_DET_BACKWARD=1` 只关闭 RMSNorm 反向的重定向，
RMSNorm 反向回到 CANN 内置算子。FA 反向已经编进 vendor，注册即生效，没有单独的
运行时开关。

## 支持范围

**RMSNorm 反向**：隐藏维 `d` 为 1…65536 任意取值；输入 rank ≥ 1（按最后一维展平成
`m x d`）；`x`/`dy` 支持 fp16 / bf16 / fp32，`gamma` 与 `x` 同 dtype 或 fp32。
不支持的组合回落到内置算子，并计数告警（`det_rms_norm.stats()`）。

**FA 反向**：`npu_fusion_attention` 能表达的调用都走确定性反向——BNSD/BSND/BSH/SBH/TND
布局、`d` 64/128/192/256/512/768、fp16/bf16/fp32、GQA、`sparse_mode` 0/2/3/4、
PSE/ALiBi、`dv != d`、varlen。MLA/`query_rope` 不在范围内（`npu_fusion_attention`
没有 rope 入参，本仓无法触达）。

## 验收

- RMSNorm 反向：`torch_custom/fla_npu/test/test_det_rms_norm_shapes.py`，192 个形状
  组合（含 fp64 参考与 A/A 逐位比对），全部通过、无回落。
- FA 反向：同一输入重复 20 次反向，`dq`/`dk`/`dv` 逐位相同。
- 端到端：固定输入的 16 rank A/A 训练，16 步内 loss 与 grad_norm 差异全为 0。
