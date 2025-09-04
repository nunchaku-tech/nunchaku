Writing Docstrings
==================

Use this guide to write clear, consistent Python docstrings for Nunchaku.
Follow the `NumPy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`__, and always specify variable shapes, dtypes, and notation.
The docstring should be concise and informative.

Docstring Format
----------------

A standard docstring should look like:

.. code-block:: python

    """
    Short summary of what the function or class does.

    (Optional) Extended description.

    Parameters
    ----------
    param1 : type
        Description.
    param2 : type, optional
        Description. Default is ...
    param3 : array-like, shape (n, m), dtype float
        Example of shape and dtype notation.

    Returns
    -------
    out1 : type
        Description.
    out2 : type
        Description.

    Raises
    ------
    ValueError
        When this exception is raised.

    See Also
    --------
    other_function : brief description

    Notes
    -----
    Additional details or references.

    Examples
    --------
    >>> result = func(1, 2)
    >>> print(result)
    3
    """

Guidelines
----------

- Use triple double quotes (`"""`) for all docstrings.
- Every public module, class, method, and function must have a docstring.
- The first line is a concise summary.
- Use sections in this order (as needed): `Parameters`, `Returns`, `Raises`, `See Also`, `Notes`, `Examples`.

Shapes, Dtypes, and Notation
----------------------------

- Always specify expected shape and dtype for tensors/arrays.
- Use plain text for shapes (not LaTeX/math symbols).
- Use clear, single-letter or descriptive names for shape dimensions (e.g., `B` for batch size).
- Define all shape symbols in a `Notes` section.

**Example:**

.. code-block:: python

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W), dtype float32
            Input image tensor.
        mask : Optional[torch.Tensor], shape (B, 1, H, W), dtype bool
            Optional mask.

        Returns
        -------
        out : torch.Tensor, shape (B, num_classes), dtype float32
            Output logits.

        Raises
        ------
        ValueError
            If input shapes are incompatible.

        Notes
        -----
        Notations:
        - B: batch size
        - C: channels
        - H: height
        - W: width
        - num_classes: number of output classes

        Examples
        --------
        >>> x = torch.randn(8, 3, 224, 224)
        >>> out = model.forward(x)
        """
        ...

Best Practices
--------------

- **Be concise and clear.** The summary should state what the function/class does.
- **Document all arguments and return values.** State if a parameter can be `None`.
- **Use `Examples`** to show typical usage.
- **Use `Raises`** to list all possible exceptions.
- **Use `Notes`** to clarify shape symbols or special behaviors.
- **Use `See Also`** for related functions or methods.

Examples
--------

.. code-block:: python

    def add(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Elementwise sum of two tensors.

        Parameters
        ----------
        a : torch.Tensor, shape (B, D), dtype float32
            First input.
        b : torch.Tensor, shape (B, D), dtype float32
            Second input.

        Returns
        -------
        out : torch.Tensor, shape (B, D), dtype float32
            Elementwise sum.

        Raises
        ------
        ValueError
            If input shapes do not match.

        Notes
        -----
        Notations:
        - B: batch size
        - D: feature dimension

        Examples
        --------
        >>> a = torch.ones(2, 3)
        >>> b = torch.zeros(2, 3)
        >>> add(a, b)
        tensor([[1., 1., 1.],
                [1., 1., 1.]])
        """

    class MyModel(nn.Module):
        """
        Example model.

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

For questions or formatting help, see existing Nunchaku code or ask in the dev chat.
