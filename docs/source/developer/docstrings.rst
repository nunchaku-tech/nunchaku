Writing Docstrings
==================

This document describes how to write Python docstrings for Nunchaku.
Nunchaku follows the **NumPy style guide** for docstrings, with additional conventions for specifying variable shapes, dtypes, and notation.

General Guidelines
------------------

- Use triple double quotes (`"""`) for all docstrings.
- Every public module, class, method, and function should have a docstring.
- The first line should be a short summary of the function/class/module.
- Use sections in the following order (as needed): `Parameters`, `Returns`, `Raises`, `Examples`, `Notes`.

Shape, Dtype, and Variable Notation
-----------------------------------

When documenting function or method parameters and return values, **always specify the expected shape and dtype** of tensors or arrays. Use plain text for shapes (not LaTeX or math symbols), and define all shape symbols in a `Notes` section.

**How to specify shapes and dtypes:**

- In the `Parameters` and `Returns` sections, after the type, add `shape (...)` and `dtype ...` as appropriate.
- Use clear, single-letter or descriptive variable names for shape dimensions (e.g., `B` for batch size, `C` for channels, `H` for height, `W` for width).
- Define all shape symbols in a `Notes` section at the end of the docstring.

**Example:**

.. code-block:: python

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Applies the model forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W), dtype float32
            Input image tensor.
        mask : Optional[torch.Tensor], shape (B, 1, H, W), dtype bool
            Optional mask tensor.

        Returns
        -------
        out : torch.Tensor, shape (B, num_classes), dtype float32
            Output logits.

        Notes
        -----
        B: batch size
        C: number of channels
        H: image height
        W: image width
        num_classes: number of output classes
        """
        ...

Best Practices
--------------

- **Be concise but informative.** The summary line should state what the function/class does, not how.
- **Document all arguments and return values.** If a parameter can be `None`, state so.
- **Use the `Examples` section** to show typical usage, especially for public APIs.
- **Use the `Raises` section** to document all exceptions that may be raised.
- **Use the `Notes` section** to clarify shape symbols, special behaviors, or implementation details.

Examples
--------

.. code-block:: python

    def add(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adds two tensors elementwise.

        Parameters
        ----------
        a : torch.Tensor, shape (B, D), dtype float32
            First input tensor.
        b : torch.Tensor, shape (B, D), dtype float32
            Second input tensor.

        Returns
        -------
        out : torch.Tensor, shape (B, D), dtype float32
            Elementwise sum of `a` and `b`.

        Notes
        -----
        B: batch size
        D: feature dimension
        """

    class MyModel(nn.Module):
        """
        Example model for demonstration.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        output_dim : int
            Output feature dimension.

        Examples
        --------
        >>> model = MyModel(input_dim=128, output_dim=10)
        >>> x = torch.randn(32, 128)
        >>> y = model(x)
        """

References
----------

- NumPy docstring guide: https://numpydoc.readthedocs.io/en/latest/format.html
- PyTorch docstring conventions: https://pytorch.org/docs/stable/docstring.html

If you have questions or are unsure about formatting, refer to existing Nunchaku code or ask in the development chat.
