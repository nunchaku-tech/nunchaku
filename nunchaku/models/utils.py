from torch import nn
import torch


def fuse_linears(linears: list[nn.Linear]) -> nn.Linear:
    assert len(linears) > 0
    if len(linears) == 1:
        return linears[0]
    else:
        assert all(linear.in_features == linears[0].in_features for linear in linears)
        out_features = sum(linear.out_features for linear in linears)
        bias = all(linear.bias is not None for linear in linears)
        return nn.Linear(
            linears[0].in_features,
            out_features,
            bias=bias,
            dtype=linears[0].weight.dtype,
            device=linears[0].weight.device,
        )


class LayerOffloadHelper:
    """
    Helper class for managing GPU memory offloading and computation pipeline for transformer layers.

    This class implements a streaming approach to overlap computation with memory transfers,
    allowing efficient GPU memory management for large models.
    """

    def __init__(self, layers: list, func_compute, func_load, func_unload):
        """
        Initialize the LayerOffloadHelper.

        Parameters
        ----------
        offload : bool
            Whether to enable GPU memory offloading.
        layers : list
            List of transformer blocks/layers to process.
        func_compute : callable
            Function to compute forward pass for a layer: func_compute(layer) -> output.
        func_load : callable
            Function to load a layer to GPU: func_load(layer) -> None.
        func_unload : callable
            Function to unload a layer from GPU: func_unload(layer) -> None.
        """
        self.layers = layers
        self.func_compute = func_compute
        self.func_load = func_load
        self.func_unload = func_unload

        if offload:
            self.stream_compute = torch.cuda.Stream()
            self.stream_load = torch.cuda.Stream()
            self.event_compute_done = None
            self.event_load_done = None

    def run(self):
        """
        Execute the computation pipeline for all layers.

        This method processes all layers sequentially, with overlapping computation
        and memory transfers when offloading is enabled.
        """
        for i, layer in enumerate(self.layers):
            self._run_layer(i, layer)

        # Wait for the last computation to complete
        if self.event_compute_done:
            self.event_compute_done.synchronize()

        # Unload the last layer
        self.func_unload(self.layers[-1])

    def _run_layer(self, idx: int, layer):
        """
        Process a single layer with the appropriate computation strategy.

        Parameters
        ----------
        idx : int
            Index of the current layer.
        layer
            The layer to process.
        """
        if not self.offload:
            # Simple sequential computation without offloading
            self.func_compute(layer)
        else:
            # Overlapped computation and memory transfers using CUDA streams
            next_compute_done = torch.cuda.Event()
            next_load_done = torch.cuda.Event()

            # Compute stream: execute forward pass
            with torch.cuda.stream(self.stream_compute):
                if self.event_load_done:
                    self.stream_compute.wait_event(self.event_load_done)
                self.func_compute(layer)
                next_compute_done.record()

            # Load/unload stream: manage memory transfers
            with torch.cuda.stream(self.stream_load):
                if self.event_compute_done:
                    self.stream_load.wait_event(self.event_compute_done)
                if idx > 0:
                    self.func_unload(self.layers[idx - 1])
                if idx + 1 < len(self.layers):
                    self.func_load(self.layers[idx + 1])
                next_load_done.record()

            self.event_compute_done = next_compute_done
            self.event_load_done = next_load_done

            # WDDM workaround: synchronize compute stream
            torch.cuda.current_stream().wait_event(self.event_compute_done)
