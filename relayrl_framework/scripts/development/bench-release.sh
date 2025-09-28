#!/bin/bash

cd ..; cd ..
RUSTFLAGS="--cfg tokio_unstable" cargo bench --no-default-features --features="networks profile" -- --release --verbose