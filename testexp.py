import torch
tensor_1d = torch.tensor([1.0, 2.0, 3.0])
print("1D Tensor:", tensor_1d)
tensor_2D = torch.tensor([[1.0, 2.0],[3.0, 4.0]])
print("2D Tensor:\n", tensor_2D)

tensor_sum= tensor_1d + torch.tensor((1.0, 2.0, 3.0))
print("Sum:", tensor_sum)

tensor_mul = torch.matmul(tensor_2D,tensor_2D )
print("Matrix Multiplication:", tensor_mul)

tensor_a = torch.tensor((2.0, 4.0, 5.0, 3.0))
tensor_b = torch.tensor((7.0, 5.0, 4.0, 3.0))

# Addition
def tensor_sum():
    tensor_sum = tensor_a + tensor_b
    print("Sum:", tensor_sum)

# Subtraction
def tensor_sub():
    tensor_sub = tensor_a - tensor_b
    print("Subtraction:", tensor_sub)


# Multiplication
def tensor_mul():
    tensor_mul = tensor_a * tensor_b
    print("Multiplication:", tensor_mul)
tensor_mul()

