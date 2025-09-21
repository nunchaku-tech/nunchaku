Building the Documentation
==========================

Follow this guide to build the Nunchaku documentation locally using Sphinx.

Step 1: Set Up the Environment
------------------------------

First, ensure your environment is prepared. This process is similar to :ref:`Installation <build-from-source>`.
If you only want the documentation dependencies (and not the full development set), use the ``docs`` extra:

.. code-block:: shell

    pip install -e ".[docs]"

Step 2: Build the Documentation
-------------------------------

Navigate to the ``docs`` directory and build the HTML documentation:

.. code-block:: shell

    cd docs
    make clean
    make html

This will generate the HTML files in the ``docs/build/html`` directory.

Step 3: Preview the Documentation
---------------------------------

To view the generated documentation locally, start a simple HTTP server:

.. code-block:: shell

    cd build/html
    python -m http.server 2333

Then open your browser and go to ``http://localhost:2333`` to browse the documentation.
Feel free to change the port to any other port you prefer.

.. tip::

   If you make changes to the documentation source files, simply rerun ``make html`` to update the output.
