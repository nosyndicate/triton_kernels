
import math

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def pytorch_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def benchmark(fn, *args, **kwargs):
    ms = triton.testing.do_bench(lambda: fn(*args, **kwargs))
    return ms

@triton.jit
def _attn_fwd_kernel(
    q, k, v, output,
    q_stride_b, q_stride_heads, q_stride_seq, q_stride_head_dim,
    k_stride_b, k_stride_heads, k_stride_seq, k_stride_head_dim,
    v_stride_b, v_stride_heads, v_stride_seq, v_stride_head_dim,
    out_stride_b, out_stride_heads, out_stride_seq, out_stride_head_dim,
    batch_size: tl.constexpr, 
    num_heads: tl.constexpr, 
    seq_len: tl.constexpr, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    head_dim: tl.constexpr,
    causal: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    # Placeholder for the actual kernel implementation
    seq_idx = tl.program_id(0) 
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # load the query block for the current head and batch
    seq_offsets = seq_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    head_dim_offsets = tl.arange(0, head_dim)

    q_offsets = (
        batch_idx * q_stride_b + 
        head_idx * q_stride_heads + 
        seq_offsets[:, None] * q_stride_seq + 
        head_dim_offsets[None, :] * q_stride_head_dim
    )

    # seq dimension is variable, so we need mask on this dimension
    q_mask = seq_offsets[:, None] < seq_len
    q = tl.load(q + q_offsets, mask=q_mask, other=0.0)

    # Initialize accumulation buffers
    # m_i is the max value over all qk, similar to the larg max est value in online softmax
    # it has shape of [BLOCK_M[, since we have one of such value for each query]]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + float('-inf')
    # denom_i is the sum of exp(qk - m_i) over k, use as the denominator in softmax
    denom_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # acc is the accumulator for the output, it has shape of [BLOCK_M, head_dim]
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # offsets for loading the head dimension
    offset_head_dim = tl.arange(0, head_dim)
    # If casual, for current block, we only attend to all previous blocks and 
    # current block up to current position
    # But if it's not casual, like Bert, we attend to all positions in the current block
    end_n = seq_len if not causal else (seq_idx + 1) * BLOCK_M
    for start_n in range(0, end_n, BLOCK_N):
        # Load the current k and 
        curr_offsets = start_n + tl.arange(0, BLOCK_N)
        k_offsets = (
            batch_idx * k_stride_b + 
            head_idx * k_stride_heads + 
            curr_offsets[:, None] * k_stride_seq + 
            offset_head_dim[None, :] * k_stride_head_dim
        )

        v_offsets = (
            batch_idx * v_stride_b + 
            head_idx * v_stride_heads + 
            curr_offsets[:, None] * v_stride_seq + 
            offset_head_dim[None, :] * v_stride_head_dim
        )

        mask = curr_offsets[:, None] < seq_len
        k_curr = tl.load(k + k_offsets, mask=mask, other=0.0)
        v_curr = tl.load(v + v_offsets, mask=mask, other=0.0)

        # q has shape of [BLOCK_M, head_dim], k and v both have shape of [BLOCK_N, head_dim]
        # so qk has shape of [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k_curr), input_precision="ieee")
        qk *= softmax_scale

        # Mask out positions outside of sequence length
        qk = tl.where(curr_offsets[None, :] < seq_len, qk, float('-inf'))
        # If causal, we also need to mask out positions in the current block 
        # that are ahead of the current query position
        if causal:
            # If position of Q is greater or equal to position of K, 
            # then it's not masked, otherwise it's masked
            causal_mask = seq_offsets[:, None] >= curr_offsets[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # qk has shape of [BLOCK_M, BLOCK_N], m_i_new has shape of [BLOCK_M], so we need to do broadcast here
        p_ij = tl.exp(qk - m_i_new[:, None])
        scale = tl.exp(m_i - m_i_new)

        # update the accumulator
        # compute the numerate part of the \softmax(qk) @ v
        # qk has shape of [BLOCK_M, BLOCK_N], v has shape of [BLOCK_N, head_dim],
        #  so the output of this dot product has shape of [BLOCK_M, head_dim]
        acc = acc * scale[:, None]
        acc += tl.dot(p_ij.to(v_curr.dtype), v_curr, input_precision="ieee")

        denom_i_curr = tl.sum(p_ij, axis=1)
        denom_i = denom_i * scale + denom_i_curr
        m_i = m_i_new


    o_offsets = (
        batch_idx * out_stride_b +
        head_idx * out_stride_heads +
        seq_offsets[:, None] * out_stride_seq +
        offset_head_dim[None, :] * out_stride_head_dim
    )

    o_mask = seq_offsets[:, None] < seq_len
    output_vals = acc / denom_i[:, None]
    tl.store(output + o_offsets, output_vals, mask=o_mask)





def custom_triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q)    

    softmax_scale = 1.0 / math.sqrt(head_dim)

    BLOCK_M = 64  # number of token to process per program
    BLOCK_N = 32  # number of tokens to process inside the loop
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size, num_heads)
    _attn_fwd_kernel[grid](
        q, k, v, output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        batch_size,
        num_heads,
        seq_len,
        BLOCK_M,
        BLOCK_N,
        head_dim,
        False,
        softmax_scale,
    )

    return output

def verify_correctness(q, k, v, tol=1e-3):
    ref_out = pytorch_sdpa(q, k, v)
    # Testing the custom triton wrapper
    tri_out = custom_triton_attention(q, k, v)
    
    is_correct = torch.allclose(ref_out, tri_out, atol=tol, rtol=tol)
    if is_correct:
        print("✅ Success: Custom kernel results match PyTorch baseline.")
    else:
        max_diff = (ref_out - tri_out).abs().max().item()
        print(f"❌ Failure: Results differ. Max difference: {max_diff:.6f}")
        print("Note: This is expected until the kernel logic is implemented.")


if __name__ == "__main__":
    batch_size = 16
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

    # Warm up
    pytorch_sdpa(q, k, v)

    ms = benchmark(pytorch_sdpa, q, k, v)
    print(f"PyTorch SDPA: {ms:.2f} ms")


    verify_correctness(q, k, v)

    ms = benchmark(custom_triton_attention, q, k, v)
    print(f"Custom Triton Attention: {ms:.2f} ms")
