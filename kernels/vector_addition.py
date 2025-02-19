import torch
import triton 
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program id along the 1D grid
    pid = tl.program_id(0)
    
    # calculate the starting offset for this program
    block_start = pid * BLOCK_SIZE
    # Compute the indices for the elements processed by this program.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to avoid out-of-bound memory access.
    mask = offsets < n_elements
    # Load the corresponing elements from the input pointers.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Compute the elmentes-wise sum.
    z = x + y 
    # store the results to the output pointers
    tl.store(out_ptr + offsets, z, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Adds two vectors x and y using the triton kernel.
    """
    # Ensure the inputs are on CUDA.
    assert x.is_cuda and y.is_cuda, "Input tensor must be on CUDA device"
    # Allocate output tensor.
    out = torch.empty_like(x)
    n_elements = x.numel()
    # Define the grid: number of blocks required
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Launch the Triton kernel.
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out
    
if __name__ == "__main__":
    # Set a manual seed for reproducibility.
    torch.manual_seed(42)
    # Define the size of the vectors.
    size = 98432
    # Create two random vectors on CUDA.
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    # Compute the expected result using pytorch.
    expected = x + y
    # Compute the result using the Triton kernel.
    result = vector_add(x, y)
    # Print the outputs.
    print("Expected:", expected)
    print("Triton Kernel Result:", result)
    # Compute and print the maximum difference.
    max_diff = torch.max(torch.abs(expected - result))
    print(f"Max difference: {max_diff}")
        
    