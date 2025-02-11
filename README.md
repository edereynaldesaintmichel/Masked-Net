# Masked-Net

This repository contains the implementation of the "Masked Net" architecture, a novel neural network design described in the blog post "[Tech report] How I analyzed 337,576 financial statements".  This architecture is specifically designed to handle datasets with a significant proportion of missing or invalid values, common in financial data.

## Key Features & Mathematical Justification

The core idea behind Masked-Net is to explicitly handle missing/invalid values (represented by a specific placeholder, e.g., 0.01) and "not applicable" values (represented by 0) differently from genuine near-zero values.  This is achieved through the use of "Masked Layers".

1.  **Input Masks:**  For each input vector, two binary masks are generated: `mask_invalid` (indicating placeholder values) and `mask_zero` (indicating zero/not-applicable values).

2.  **Concatenation and Projection:** The input vector and the two masks are concatenated. A crucial optimization involves projecting the input *and* the masks into a lower-dimensional subspace *using the same projection matrix*.  This leverages the inherent similarity in how these values should be treated: as deviations from the "average" expected value. Mathematically, this is represented as:

    ```
    projected_input = main_proj @ input_values
    projected_mask_invalid = main_proj @ mask_invalid
    projected_mask_zero = main_proj @ mask_zero
    concatenated_input = concat(projected_input, projected_mask_invalid, projected_mask_zero)
    ```

    where `main_proj` is a shared, learned projection matrix.  This shared projection acts like a learned 1D convolution, extracting features relevant to both the values and their validity.

3.  **Shared Projection (Convolution Analogy):**  The use of the *same* `main_proj` matrix for all three components (values, invalid mask, zero mask) is a key innovation.  This is analogous to a 1D convolution with `out_features` filters, `in_features` stride, and a filter size of 1.  The `main_proj` matrix learns filters that effectively process both the main values and the information encoded in the masks, maximizing information retention.

4.  **LoRA for Global Interactions:**  The output of the concatenated projection is then passed through a linear layer, followed by a LeakyReLU activation.  To further enhance parameter efficiency, Low-Rank Approximation (LoRA) is applied to the linear layer's transformation matrix (`result_proj`).  This captures global feature interactions with a low-rank representation, promoting generalization and reducing overfitting. The LoRA is implemented before the activation function.

    The combination of the shared projection (convolutional aspect) and LoRA allows the network to efficiently learn both local feature representations (via the convolution-like projection) and global feature dependencies (via LoRA).

5. **Masked Layer Parameter Count**:
Each Masked Layer's parameter count is a function of the input, output and LoRA rank.

6.  **Output Layer:**  The final Masked Layer's output is fed into a standard linear layer to produce the final prediction.

## Advantages

-   **Handles Missing Data Effectively:**  The explicit masking mechanism allows the network to treat missing/invalid data as distinct from genuine near-zero values, improving prediction accuracy in sparse datasets.
-   **Parameter Efficiency:**  The shared projection matrix and LoRA significantly reduce the number of parameters compared to a standard MLP, mitigating overfitting, especially on smaller datasets.
-   **Improved Generalization:**  The architecture's design, combining convolutional principles with LoRA, encourages the learning of robust features and relationships, leading to better generalization performance.
-    **Easy to use**: the implementation is very straightforward and consists of one custom layer.

## Data
Please DM me to get the data. I can't make it available as it comes from a payed source.
