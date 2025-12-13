const { spawnSync } = require('child_process');
const fs = require('fs');

function checkRust() {
    try {
        const cargo = spawnSync('cargo', ['--version'], { encoding: 'utf8' });
        if (cargo.error || cargo.status !== 0) {
            return false;
        }
        return true;
    } catch (e) {
        return false;
    }
}

console.log('➤ numrs-node: Checking for Rust compiler...');

if (!checkRust()) {
    console.error('\n\x1b[31m✖ Rust compiler not found!\x1b[0m');
    console.error('  To install numrs-node (native), you need to have Rust installed.');
    console.error('  Please install it from \x1b[36mhttps://rustup.rs/\x1b[0m and try again.');
    console.error('  Alternatively, use @numrs/wasm for a pre-compiled, slightly slower version.\n');
    process.exit(1);
}

console.log('➤ numrs-node: Rust found! Compiling native extension (this might take a minute)...');

const build = spawnSync('npm', ['run', 'build:native'], {
    stdio: 'inherit',
    shell: true
});

if (build.status !== 0) {
    console.error('\n\x1b[31m✖ Build failed!\x1b[0m');
    console.error('  Check the compilation logs above for details.');
    process.exit(1);
}

console.log('➤ numrs-node: Native build successful! 🚀\n');
