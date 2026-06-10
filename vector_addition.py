import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(axis=0)
    block_start = program_id * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    nums_in_a = tl.load(a + offsets, mask=mask)
    nums_in_b = tl.load(b + offsets, mask=mask)

    nums_in_c  = nums_in_a + nums_in_b

    tl.store(c + offsets, nums_in_c, mask=mask)
    


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)



if __name__ == "__main__":
    N = 10_000_000
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    c = torch.empty_like(a)

    solve(a, b, c, N)

    # Verify the result
    assert torch.allclose(c, a + b)
    print("Vector addition successful!")