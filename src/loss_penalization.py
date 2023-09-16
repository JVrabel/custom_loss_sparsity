"""
Contains functions for computing a custom term to regularize the loss function. 
"""
import torch

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

import torch

def sparseloc(weight_tensor):
    total_sum = 0.0  # variable to store the total sum

    # Loop over the rows of the weight_tensor
    for i in range(weight_tensor.size(0)):
        W = weight_tensor[i, :]

        # Compute the outer product directly to make it differentiable
        W_out = torch.outer(W, W)

        # Zero out the elements above the diagonal and far from the diagonal
        lower_triangle = torch.tril(W_out, -1)
        band_matrix = torch.triu(lower_triangle, -3)

        # Sum the absolute values and add to total_sum
        total_sum += torch.sum(torch.abs(band_matrix))

    return total_sum
