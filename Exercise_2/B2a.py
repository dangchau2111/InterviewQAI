import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    # Tính khoảng cách Euclidean giữa anchor và positive
    pos_dist = np.sum((anchor - positive) ** 2)
    
    # Tính khoảng cách Euclidean giữa anchor và negative
    neg_dist = np.sum((anchor - negative) ** 2)
    
    # Tính giá trị của Triplet Loss
    loss = np.maximum(pos_dist - neg_dist + margin, 0)
    
    return loss

# Ví dụ sử dụng
anchor = np.array([1.0, 1.0])
positive = np.array([1.1, 1.1])
negative = np.array([2.0, 2.0])

# Tính Triplet Loss với margin = 1.0
loss = triplet_loss(anchor, positive, negative, margin=1.0)
print(f'Triplet Loss: {loss}')
