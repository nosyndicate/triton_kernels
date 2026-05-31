import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F

def pytorch_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def benchmark(fn, *args, **kwargs):
    ms = triton.testing.do_bench(lambda: fn(*args, **kwargs))
    return ms

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, 
    O,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    BATCH: tl.constexpr, 
    HEADS: tl.constexpr, 
    SEQ_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE_HEAD_DIM: tl.constexpr,
    IF_CAUSAL_MASK: tl.constexpr,
    softmax_scale: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_head_id = tl.program_id(1)

    batch_id = batch_head_id // HEADS
    head_id = batch_head_id % HEADS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_SIZE_HEAD_DIM)

    q_ptrs = (Q + batch_id * stride_qb + head_id * stride_qh +
              offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    q_mask = offs_m[:, None] < SEQ_LEN
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize online softmax variables
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + float('-inf')
    acc = tl.zeros([BLOCK_M, BLOCK_SIZE_HEAD_DIM], dtype=tl.float32)

    end_n = SEQ_LEN if not IF_CAUSAL_MASK else (start_m + 1) * BLOCK_M

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # K tile: (BLOCK_N, BLOCK_SIZE_HEAD_DIM). It is transposed for q @ k^T
        # below.
        k_ptrs = (K + batch_id * stride_kb + head_id * stride_kh
                   + offs_n[:,None] * stride_kn + offs_k[None,:] * stride_kk )
        #k_ptrs = (K + batch_id * stride_kz + head_id * stride_kh
                   #+  offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn ) # Swapped strides and offsets

        # V pointers: (BLOCK_SIZE_N, BLOCK_SIZE_HEAD_DIM)
        v_ptrs = (V + batch_id * stride_vb + head_id * stride_vh +
                  offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

        # Mask for K and V loading
        k_mask = offs_n[:, None] < SEQ_LEN
        #k_mask=  offs_n[None,:] <N_CTX # Mask applies to the dimension varying with 'n'
        # V mask
        v_mask = offs_n[:, None] < SEQ_LEN

        # Load K tile (BLOCK_N, BLOCK_SIZE_HEAD_DIM)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        # Load V tile (shape will be BLOCK_SIZE_N x BLOCK_SIZE_HEAD_DIM)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)


        # Compute attention scores: Q (M, K) * K^T (K, N) -> (M, N)
        #tl.debug_barrier()
        #tl.print(“qk tile =”, qk)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")
        qk*=  softmax_scale
        qk = tl.where(offs_n[None, :] < SEQ_LEN, qk, float('-inf'))

        # Apply causal mask if needed
        if IF_CAUSAL_MASK:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Online softmax computation
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p_ij = tl.exp(qk - m_i_new[:, None])
        scale = tl.exp(m_i - m_i_new)

        # Update accumulator
        acc = acc * scale[:, None]
        # p_ij (M, N), v (N, K) -> dot (M, K)
        acc += tl.dot(p_ij.to(v.dtype), v, input_precision="ieee") # Removed trans_b=True, dot(A, B) expects B with columns matching A rows

        # Update normalizing factors
        l_i_current = tl.sum(p_ij, axis=1)
        l_i = l_i * scale + l_i_current
        m_i = m_i_new

    # Store outputs
    O_ptrs = (O + batch_id * stride_ob + head_id * stride_oh +
              offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)

    acc_o = acc / l_i[:, None]
    o_mask = offs_m[:, None] < SEQ_LEN

    tl.store(O_ptrs, acc_o, mask=o_mask)
    

def custom_triton_attention(q, k, v):
    batch_size, num_heads, seq_len, head_dim = q.shape
    if head_dim not in (16, 32, 64, 128):
        raise ValueError("custom_triton_attention only supports head_dim in {16, 32, 64, 128}")

    output = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(seq_len, BLOCK_M) ,batch_size * num_heads)

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
