"""
Contains functions for computing a custom term to regularize the loss function. 
"""

import torch
import torch.nn.functional as F

# def sparseloc(weight_tensor):
#     total_sum = 0.0  # variable to store the total sum

#     # Initialize W_out to zeros to save memory
#     W_out = torch.zeros(weight_tensor[0,:].size(0), weight_tensor[0,:].size(0))

#     # Loop over the rows of the weight_tensor
#     for i in range(weight_tensor.size(0)):
#         W = weight_tensor[i, :]

#         # Compute the outer product directly to make it differentiable
#         W_out = torch.outer(W, W)

#         # Zero out the elements above the diagonal and far from the diagonal
#         W_out.tril_(-1)
#         W_out.triu_(-4)

#         # Sum the absolute values and add to total_sum
#         total_sum += torch.sum(torch.abs(W_out))

#     return total_sum


# def sparseloc(weight_tensor):
#     total_sum = 0.0  # variable to store the total sum

#     # Loop over the rows of the weight_tensor
#     for i in range(weight_tensor.size(0)):
#         W = weight_tensor[i, :]

#         # Compute the outer product directly to make it differentiable
#         W_out = torch.outer(W, W)

#         # Zero out the elements above the diagonal and far from the diagonal
#         W_out = torch.tril(W_out, -1)
#         W_out = torch.triu(W_out, -5)
#         # lower_triangle = torch.tril(W_out, -1)
#         # band_matrix = torch.triu(lower_triangle, -5)
#         # del W, W_out, lower_triangle
#         # Sum the absolute values and add to total_sum
#         total_sum += torch.sum(torch.abs(W_out))
#         # total_sum += torch.sum(torch.abs(band_matrix))

#     return total_sum

# import random

def sparseloc(weight_tensor, num_samples=1000):
    total_sum = torch.tensor(0.0, dtype=weight_tensor.dtype, device=weight_tensor.device)
    
    n_rows, n_cols = weight_tensor.shape

    # Randomly sample 'num_samples' rows to consider
    sampled_rows = random.sample(range(n_rows), min(num_samples, n_rows))

    for i in sampled_rows:
        W = weight_tensor[i, :]

        # Randomly sample pairs of elements within this row
        sampled_pairs = random.sample(range(n_cols), 2)

        # Calculate the product of the sampled elements
        product = W[sampled_pairs[0]] * W[sampled_pairs[1]]

        # Increment the total sum
        total_sum += torch.abs(product)

    return total_sum


# Function to perform 2D convolution using a custom filter
def multipl_1d_sliding_window(weight_tensor, filter_values):
    filter_values = filter_values.to(weight_tensor.device)
    sum_filter = torch.tensor(0.0, dtype=weight_tensor.dtype, device=weight_tensor.device)
    


    # Unfolding
    unfolded_input = F.unfold(weight_tensor.unsqueeze(0).unsqueeze(0), filter_values.shape, stride=1, padding=0)
    unfolded_input = unfolded_input.transpose(1, 2).contiguous().view(-1, unfolded_input.size(1))
    mask = ~filter_values.eq(0.0).squeeze() # mask for zero values in the filter
    # Perform element-wise multiplication followed by multiplication of masked intermediate results

    masked_unfolded_input = unfolded_input[:,mask]
    masked_filter_values = filter_values[:,mask]

    output_unfolded = (masked_unfolded_input * masked_filter_values).prod(dim=1)
    sum_filter = output_unfolded.sum()
    
    # # Fold the result back
    # output_shape = (weight_tensor.size(0) - filter_values.size(0) + 1, weight_tensor.size(1) - filter_values.size(1) + 1)
    # folded_output = F.fold(output_unfolded.view(1, 1, -1), output_shape, kernel_size=(1, 1), stride=1)
    
    # return folded_output.squeeze()
    return sum_filter

def sparseloc_filter(weight_tensor):
    weight_tensor = weight_tensor.abs()
    filter2 = torch.tensor([1.0, 1.0], dtype=torch.float32).view(1, -1)
    filter3 = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32).view(1, -1)
    filter4 = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32).view(1, -1)
    filter5 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32).view(1, -1)
    filter6 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32).view(1, -1)
    filter7 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  1.0], dtype=torch.float32).view(1, -1)
    filter8 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32).view(1, -1)
    filter9 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32).view(1, -1)
    result = (multipl_1d_sliding_window(weight_tensor, filter2) +
              multipl_1d_sliding_window(weight_tensor, filter3) +
              multipl_1d_sliding_window(weight_tensor, filter4) +
              multipl_1d_sliding_window(weight_tensor, filter5) +   
              multipl_1d_sliding_window(weight_tensor, filter6) +   
              multipl_1d_sliding_window(weight_tensor, filter7) +   
              multipl_1d_sliding_window(weight_tensor, filter8) +   
              multipl_1d_sliding_window(weight_tensor, filter9)    
            )
    return result



def calculate_sparsity(tensor):
    # Flatten the tensor
    flat_tensor = tensor.view(-1)
    
    # Calculate the 5% quantile
    q_5 = torch.quantile(torch.abs(flat_tensor), 0.05)
    
    # Count elements above the quantile in absolute value
    count_non_zero = torch.sum(torch.abs(flat_tensor) > q_5).item()
    
    # Calculate the total number of elements
    total_elements = flat_tensor.numel()
    
    # Calculate sparsity
    sparsity = 1 - (count_non_zero / total_elements)
    
    return sparsity





 