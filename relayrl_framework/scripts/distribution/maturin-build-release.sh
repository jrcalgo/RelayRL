#!/bin/bash
set -e
set -u
set -x

MANIFEST_DIR="rust/relayrl_framework/Cargo.toml"

# Format the code
cargo fmt --all --manifest-path "$MANIFEST_DIR"

## Clean previous builds
#cargo clean --manifest-path "$MANIFEST_DIR"

# Build the wheel using maturin
maturin build -r --manifest-path "$MANIFEST_DIR"

# Capture the original wheel filename into a variable
WHEEL_FILE=$(ls rust/relayrl_framework/target/wheels/relayrl_framework*.whl)

# Unzip the wheel into a temporary directory
unzip "$WHEEL_FILE" -d temp_wheel_dir

# Find the .so file (for verification)
if [ -z "$(find temp_wheel_dir -name "*.so")" ]; then
    echo "No shared object files found in the wheel."
    exit 1
fi

# Apply permissions to wheel
chmod 777 temp_wheel_dir/relayrl_framework/relayrl_framework*.so

# Identify the correct libtorch lib directory
LIBTORCH_LIB_DIR=$(for dir in rust/relayrl_framework/target/release/build/torch-sys*/out/libtorch/libtorch/lib; do
    if [ -d "$dir" ]; then
        echo "$dir"
        break
    fi
done)

# Create the destination directory in the package for libtorch libraries
mkdir -p temp_wheel_dir/relayrl_framework/libtorch/lib

# Copy libtorch shared libraries and __init__ into the package
cp "$LIBTORCH_LIB_DIR"/*.so temp_wheel_dir/relayrl_framework/libtorch/lib/
cp "$LIBTORCH_LIB_DIR"/*.so.1 temp_wheel_dir/relayrl_framework/libtorch/lib/
cp rust/relayrl_framework/__init__.py temp_wheel_dir/relayrl_framework/

# Set RPATH to $ORIGIN/libtorch/lib for portability
patchelf --set-rpath '$ORIGIN/libtorch/lib' temp_wheel_dir/relayrl_framework/relayrl_framework*.so

# Repackage the wheel with the original name
cd temp_wheel_dir
zip -r "../$(basename "$WHEEL_FILE")" .
cd ..

# Clean up temporary directory
rm -rf temp_wheel_dir

# Reinstall the package
pip uninstall -y relayrl_framework
pip install "$(basename "$WHEEL_FILE")"

exit 0
