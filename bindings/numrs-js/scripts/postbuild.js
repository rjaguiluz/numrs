const fs = require('fs');
const path = require('path');

const release = path.join(__dirname, '..', 'target', 'release');
const dest = path.join(release, 'numrs_node.node');

// Platform-specific library names and extensions
// Unix-like systems add 'lib' prefix, Windows does not
const platformFiles = {
  'win32': ['numrs_node.dll', 'numrs_node.node'],
  'darwin': ['libnumrs_node.dylib', 'numrs_node.dylib', 'numrs_node.node'],
  'linux': ['libnumrs_node.so', 'numrs_node.so', 'numrs_node.node']
};

const filesToTry = platformFiles[process.platform] || ['numrs_node.node', 'numrs_node.dll', 'libnumrs_node.so', 'numrs_node.so', 'libnumrs_node.dylib', 'numrs_node.dylib'];

// Find the native library
let found = false;
for (const filename of filesToTry) {
  const src = path.join(release, filename);
  if (fs.existsSync(src)) {
    try {
      // Always copy/overwrite to ensure latest build
      fs.copyFileSync(src, dest);
      console.log(`postbuild: copied ${filename} -> numrs_node.node`);
      found = true;
      break;
    } catch (e) {
      console.warn('postbuild: copy failed', e.message);
    }
  }
}

if (!found) {
  console.error('postbuild: no native library found in target/release/');
  console.error('Looking for one of:', filesToTry.join(', '));
  process.exit(1);
}
