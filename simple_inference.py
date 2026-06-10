import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def _kernel(
    input_ptr, w_ptr, b_ptr, output_ptr,
    batch_size, input_size, output_size,
    TILE_B: tl.constexpr, TILE_I: tl.constexpr, TILE_O: tl.constexpr
):
    b_idx = tl.program_id(0)
    o_idx = tl.program_id(1)

    b_offsets = b_idx * TILE_B + tl.arange(0, TILE_B)
    o_offsets = o_idx * TILE_O + tl.arange(0, TILE_O)


    acc = tl.zeros([TILE_B, TILE_O], dtype=tl.float32)

    for start_pos in range(0, input_size, TILE_I):
        i_curr_offset = start_pos + tl.arange(0, TILE_I)

        input_offsets = b_offsets[:, None] * input_size + i_curr_offset[None, :]
        w_offsets = o_offsets[:, None] * input_size + i_curr_offset[None, :]

        input_mask = (b_offsets[:, None] < batch_size) & (i_curr_offset[None, :] < input_size)
        w_mask = (o_offsets[:, None] < output_size) & (i_curr_offset[None, :] < input_size)

        input_value = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0).to(tl.float32)  # (TILE_B, TILE_I)
        w_value = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32) # (TILE_O, TILE_I)

        acc += tl.dot(input_value, tl.trans(w_value), input_precision="ieee")  # (TILE_B, TILE_O)


    bias_mask = o_offsets < output_size
    bias = tl.load(b_ptr + o_offsets, mask=bias_mask, other=0.0).to(tl.float32)  # (TILE_O,)
    acc += bias[None, :]

    output_offsets = b_offsets[:, None] * output_size + o_offsets[None, :]
    output_mask = (b_offsets[:, None] < batch_size) & (o_offsets[None, :] < output_size)

    tl.store(output_ptr + output_offsets, acc, mask=output_mask)



def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    bias = model.bias
    weight = model.weight 

    batch_size, input_size = input.shape
    _, output_size = output.shape

    TILE_B, TILE_I, TILE_O = 16, 16, 16

    grid = (triton.cdiv(batch_size, TILE_B), triton.cdiv(output_size, TILE_O))
    _kernel[grid](
        input,  # (batch, input_size) 
        weight,   # (output_size, input_size)
        bias,   # (output_size,)
        output,  # (batch, output_size)
        batch_size,
        input_size, 
        output_size,
        TILE_B, TILE_I, TILE_O
    )



if __name__ == "__main__":
    batch_size = 4
    input_size = 8
    output_size = 16

    input_tensor = torch.randn(batch_size, input_size, device='cuda')
    layer = nn.Linear(input_size, output_size).to('cuda')

    output = torch.empty(batch_size, output_size, device='cuda')

    solve(input_tensor, layer, output)
    print("Output from custom Triton kernel: ", output)
    print("Output from nn.Linear layer: ", layer(input_tensor))

    assert torch.allclose(output, layer(input_tensor)), "Output does not match expected result from nn.Linear layer."