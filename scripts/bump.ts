
import { spawn } from "bun";
import path from "path";
import fs from "fs";

const CWD = path.dirname(import.meta.dir); // Parent of scripts/ is root

// Colors for console output
const colors = {
    reset: "\x1b[0m",
    bright: "\x1b[1m",
    red: "\x1b[31m",
    green: "\x1b[32m",
    yellow: "\x1b[33m",
    cyan: "\x1b[36m",
    magenta: "\x1b[35m",
};

async function updateFile(filePath: string, replacer: (content: string) => string) {
    const file = Bun.file(filePath);
    if (!await file.exists()) {
        console.error(`${colors.red}[Error]${colors.reset} File not found: ${filePath}`);
        return;
    }
    const content = await file.text();
    const newContent = replacer(content);
    if (content !== newContent) {
        await Bun.write(filePath, newContent);
        console.log(`Updated ${path.basename(filePath)}`);
    } else {
        console.log(`No changes needed for ${path.basename(filePath)}`);
    }
}

async function promptVersion(): Promise<string> {
    process.stdout.write(`${colors.bright}Enter the new version for all packages (e.g. 0.1.0): ${colors.reset}`);
    for await (const line of console) {
        const v = line.trim();
        if (v) return v;
    }
    return "";
}

async function main() {
    console.log(`${colors.bright}${colors.magenta}NumRs Version Bumper${colors.reset}`);

    // 0. Get Version
    const newVersion = await promptVersion();
    if (!newVersion) {
        console.error("No version provided. Exiting.");
        process.exit(1);
    }
    console.log(`${colors.yellow}Updating all manifests to: ${newVersion}${colors.reset}\n`);

    // Paths
    const corePath = path.join(CWD, "numrs-core");
    const cPath = path.join(CWD, "bindings", "numrs-c");
    const jsPath = path.join(CWD, "bindings", "numrs-js");
    const wasmPath = path.join(CWD, "bindings", "numrs-wasm");
    const pyPath = path.join(CWD, "bindings", "numrs-py");
    const rPath = path.join(CWD, "bindings", "numrs-r");

    // 1. Update Versions in Files

    // numrs-core/Cargo.toml
    await updateFile(path.join(corePath, "Cargo.toml"), (c) => {
        return c.replace(/^version = ".*"/m, `version = "${newVersion}"`);
    });

    // numrs-c/Cargo.toml
    // Updates package version and numrs-core dependency version
    await updateFile(path.join(cPath, "Cargo.toml"), (c) => {
        let newC = c.replace(/^version = ".*"/m, `version = "${newVersion}"`);
        // Ensure numrs dependency has version for crates.io
        // Pattern: numrs = { path = "...", ... }
        // We expect: numrs = { path = "../numrs-core", version = "NEW_VER", ... }
        const numrsDepRegex = /^numrs = \{([^}]*)\}/m;
        const match = newC.match(numrsDepRegex);
        if (match) {
            let content = match[1];
            if (content.includes("version =")) {
                content = content.replace(/version = "[^"]*"/, `version = "${newVersion}"`);
            } else {
                // Insert version after path if not present (only if strictly needed, but let's assume consistence)
                // If it was path-only, we add version. 
                // However, crates.io publishing requires version.
                content = content.replace(/path = "[^"]*"/, `$&, version = "${newVersion}"`);
            }
            newC = newC.replace(numrsDepRegex, `numrs = {${content}}`);
        }
        return newC;
    });

    // bindings/numrs-js/package.json
    await updateFile(path.join(jsPath, "package.json"), (c) => {
        return c.replace(/"version": ".*"/, `"version": "${newVersion}"`);
    });

    // bindings/numrs-wasm/package.json
    await updateFile(path.join(wasmPath, "package.json"), (c) => {
        return c.replace(/"version": ".*"/, `"version": "${newVersion}"`);
    });

    // bindings/numrs-py/setup.py
    await updateFile(path.join(pyPath, "setup.py"), (c) => {
        return c.replace(/version=".*"/, `version="${newVersion}"`);
    });

    // bindings/numrs-r/DESCRIPTION
    await updateFile(path.join(rPath, "DESCRIPTION"), (c) => {
        return c.replace(/^Version: .*/m, `Version: ${newVersion}`);
    });

    console.log(`\n${colors.green}[Success] Version updates complete.${colors.reset}`);
    console.log(`\nNow run:\n  git commit -am "chore: bump version to ${newVersion}"\n  git tag v${newVersion}\n  git push --tags`);
}

main();
