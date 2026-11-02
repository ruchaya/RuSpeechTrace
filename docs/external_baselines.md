# External baseline: AASIST

Official source: https://github.com/clovaai/aasist

The official `Model` uses `self.out_layer = nn.Linear(5 * gat_dims[1], 2)`.
The local adapter loads that implementation at runtime and replaces only the final layer with `num_classes` outputs.

The third-party source is not redistributed in this archive.
