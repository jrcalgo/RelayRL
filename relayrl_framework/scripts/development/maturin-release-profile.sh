#!/bin/bash

# Ensure your python installation is activated/maturin is installed
cd ..; cd ..
RUSTFLAGS="--cfg=tokio_unstable" maturin develop --features profile --release