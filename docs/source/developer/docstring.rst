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

- Use triple double quotes (``"""``) for all docstrings.
- Every public module, class, method, and function must have a docstring.
- The first line is a concise summary.
- Use sections in this order (as needed): ``Parameters``, ``Returns``, ``Raises``, ``See Also``, ``Notes``, ``Examples``.

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

Writing Tips
~~~~~~~~~~~~

- **Be concise and clear.** Start with a short summary describing what the function or class does.
- **Document all parameters and return values.** Indicate if a parameter can be `None`.
- **Include an** ``Examples`` **section** to demonstrate typical usage.
- **List all possible exceptions in a** ``Raises`` **section.**
- **Use a** ``Notes`` **section** to define shape symbols and explain special behaviors.
- **Add a** ``See Also`` **section** for related functions or methods.

Useful Prompts
~~~~~~~~~~~~~~

.. code-block:: text

   Improve the writing of the docstring according to this guide. Be concise. Organize my comments clearly.

   Write the docstring for this module (every class and functions) according to this guide. Each function and class that can be shwon properly and beautifully in sphinx html. Be concise. Organize my comments clearly.

For further questions or formatting help, refer to existing Nunchaku code or ask in the developer chat.
