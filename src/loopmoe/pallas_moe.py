"""Custom Mosaic TPU Pallas kernels for MoE token dispatch (pipelined async DMA) and fused K-way combine."""

from __future__ import annotations

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


def can_use_pallas_moe(n_rows: int, m_rows: int, d_model: int, top_k: int) -> bool:
    """Return True when running on TPU with shapes matching Mosaic TPU DMA tile alignment."""
    return (
        jax.default_backend() == "tpu"
        and d_model % 1024 == 0
        and n_rows % 128 == 0
        and m_rows % 128 == 0
        and top_k >= 1
    )


def pallas_moe_gather(x_2d: jax.Array, idx_1d: jax.Array, interpret: bool = False) -> jax.Array:
    """Gather rows `[idx_1d]` from `x_2d` [N, D] via pipelined TPU DMA onto exact (8, D//8) tiles.

    Reshaping [N, D] -> [N, 8, D//8] puts each token row into its own contiguous 4 KB tile group
    so Mosaic DMA eliminates the 8x sublane read amplification of [N, D] XLA dynamic gather.
    """
    n_rows, d_model = x_2d.shape
    m_rows = idx_1d.shape[0]
    block_m = 256 if (m_rows % 256 == 0) else 128
    x_3d = x_2d.reshape(n_rows, 8, d_model // 8)

    def kernel(x_hbm, idx_smem, out_vmem, sem):
        def start_body(i, _):
            r = idx_smem[i]
            pltpu.make_async_copy(x_hbm.at[r], out_vmem.at[i], sem).start()

        jax.lax.fori_loop(0, block_m, start_body, None)
        pltpu.make_async_copy(out_vmem, out_vmem, sem).wait()

    out_3d = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m_rows, 8, d_model // 8), x_2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(m_rows // block_m,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((block_m,), lambda i: (i,), memory_space=pltpu.SMEM),
            ],
            out_specs=pl.BlockSpec(
                (block_m, 8, d_model // 8), lambda i: (i, 0, 0), memory_space=pltpu.VMEM
            ),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="pallas_moe_gather",
    )(x_3d, idx_1d)
    return out_3d.reshape(m_rows, d_model)


def pallas_moe_combine(
    down_2d: jax.Array,
    inv_order: jax.Array,
    top_k: int,
    interpret: bool = False,
) -> jax.Array:
    """Gather `top_k` expert outputs per token with double-buffered DMA and sum in VMEM.

    Eliminates intermediate partial-sum HBM roundtrips and reduces K separate [M, D] gathers
    into one fused VMEM accumulation kernel.
    """
    m_rows, d_model = down_2d.shape
    n_rows = m_rows // top_k
    block_n = 256 if (n_rows % 256 == 0) else 128
    k_pad = ((top_k + 7) // 8) * 8

    down_3d = down_2d.reshape(m_rows, 8, d_model // 8)
    inv_kn = inv_order.reshape(n_rows, top_k).T  # [K, N]
    if k_pad > top_k:
        inv_kn = jnp.pad(inv_kn, ((0, k_pad - top_k), (0, 0)))

    def kernel(down_hbm, inv_smem, out_vmem, buf_vmem, sems):
        def _start_k(k_idx: int, dst_ref, sem):
            def body(i, _):
                r = inv_smem[k_idx, i]
                pltpu.make_async_copy(down_hbm.at[r], dst_ref.at[i], sem).start()

            jax.lax.fori_loop(0, block_n, body, None)

        def _wait_buf(dst_ref, sem):
            pltpu.make_async_copy(dst_ref, dst_ref, sem).wait()

        _start_k(0, buf_vmem.at[0], sems.at[0])
        if top_k > 1:
            _start_k(1, buf_vmem.at[1], sems.at[1])
        _wait_buf(buf_vmem.at[0], sems.at[0])
        acc = buf_vmem[0]
        for k in range(1, top_k):
            cur = k % 2
            nxt = (k + 1) % 2
            if k + 1 < top_k:
                _start_k(k + 1, buf_vmem.at[nxt], sems.at[nxt])
            _wait_buf(buf_vmem.at[cur], sems.at[cur])
            acc = acc + buf_vmem[cur]
        out_vmem[...] = acc

    out_3d = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows, 8, d_model // 8), down_2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(n_rows // block_n,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((k_pad, block_n), lambda i: (0, i), memory_space=pltpu.SMEM),
            ],
            out_specs=pl.BlockSpec(
                (block_n, 8, d_model // 8), lambda i: (i, 0, 0), memory_space=pltpu.VMEM
            ),
            scratch_shapes=[
                pltpu.VMEM((2, block_n, 8, d_model // 8), down_2d.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="pallas_moe_combine",
    )(down_3d, inv_kn)
    return out_3d.reshape(n_rows, d_model)
