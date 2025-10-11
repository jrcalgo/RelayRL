#!/bin/bash

cd ..; cd ..
RUSTFLAGS="--cfg tokio_unstable" cargo build --no-default-features --features="networks profile" --release