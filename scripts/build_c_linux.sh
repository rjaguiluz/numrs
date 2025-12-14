#!/bin/sh
set -xe

echo "Starting Linux C Build Script"
pwd
ls -la

# Install build dependencies
if [ -f /etc/alpine-release ]; then
    echo "Detected Alpine Linux"
    apk add --no-cache curl build-base
    # Musl defaults to static linking. Force dynamic linking for cdylib.
    export RUSTFLAGS="-C target-feature=-crt-static"
else
    echo "Detected Manylinux (CentOS/RHEL)"
    yum install -y curl || true
fi

# Install Rust
echo "Installing Rust..."
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# Build C bindings
echo "Building numrs-c..."
export CARGO_TARGET_DIR=/tmp/numrs-target
# Use absolute path to be safe, assuming project is at /project
cargo build --release --manifest-path /project/bindings/numrs-c/Cargo.toml

# Debug and Install
echo "Build finished, checking output..."
ls -la /tmp/numrs-target/release/

echo "Installing libnumrs_c.so to /usr/lib/"
cp /tmp/numrs-target/release/libnumrs_c.so /usr/lib/
ls -l /usr/lib/libnumrs_c.so

echo "Linux C Build Script Completed Successfully"
