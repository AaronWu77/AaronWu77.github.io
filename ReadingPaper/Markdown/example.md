# Attention Is All You Need (Note)

**Author:** Vaswani et al.  
**Date:** 2017

## 1. Introduction

This paper introduces the **Transformer** model, which is solely based on attention mechanisms, dispensing with recurrence and convolutions entirely.

> "To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution."

## 2. Model Architecture

The model follows an encoder-decoder structure.

### 2.1 Encoder and Decoder Stacks
*   **Encoder**: Composed of a stack of $N=6$ identical layers.
*   **Decoder**: Also composed of a stack of $N=6$ identical layers.

### 2.2 Attention Mechanism

The function can be described as mapping a query and a set of key-value pairs to an output.

**Scaled Dot-Product Attention:**

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

Where:
*   $Q$: Queries
*   $K$: Keys
*   $V$: Values

## 3. Why Self-Attention?

There are three main reasons for using self-attention:
1.  Total computational complexity per layer.
2.  Amount of computation that can be parallelized.
3.  Path length between long-range dependencies in the network.

## 4. Code Example (PyTorch style)

```python
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) /  math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)
```

## 5. Conclusion

The Transformer is the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

---
*Note: This page is loaded from an external 'example.md' file.*