import torch

# requires_grad 是否参与梯度计算

x = torch.randn(5, 5)
y = torch.randn(5, 5)
z = torch.randn((5, 5), requires_grad=True)
a = x + y
print(a.requires_grad)
b = a + z
print(b.requires_grad)
