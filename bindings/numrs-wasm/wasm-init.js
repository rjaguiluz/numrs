// ============================================================================
// WASM Initialization with Custom Memory
// ============================================================================
//
// Inicializa el módulo WASM con memoria configurada (256 páginas inicial, 4GB máx)

const fs = require('fs');
const path = require('path');

// Configuración de memoria WASM
const INITIAL_PAGES = 256;  // 16MB inicial
const MAXIMUM_PAGES = 65536; // 4GB máximo (páginas de 64KB)

async function initWasm() {
    // Cargar el binario WASM
    const wasmPath = path.join(__dirname, 'pkg', 'numrs_wasm_bg.wasm');
    const wasmBuffer = fs.readFileSync(wasmPath);

    // Crear memoria compartida con configuración optimizada
    const memory = new WebAssembly.Memory({
        initial: INITIAL_PAGES,
        maximum: MAXIMUM_PAGES,
        shared: false  // Cambiar a true si se usan threads
    });

    // Importar el módulo con la memoria custom
    const wasmModule = await WebAssembly.compile(wasmBuffer);
    
    // Los imports que espera wasm-bindgen
    const imports = {
        './numrs_wasm_bg.js': require('./pkg/numrs_wasm_bg.js'),
        env: {
            memory
        },
        wbg: {
            __wbindgen_throw: (ptr, len) => {
                throw new Error(getString(ptr, len));
            }
        }
    };

    const instance = await WebAssembly.instantiate(wasmModule, imports);

    // Exportar funciones y memoria
    return {
        ...instance.exports,
        memory,
        getMemoryInfo: () => ({
            initial: INITIAL_PAGES,
            current: memory.buffer.byteLength / 65536,
            maximum: MAXIMUM_PAGES,
            size_mb: memory.buffer.byteLength / (1024 * 1024)
        })
    };
}

// Helper para leer strings desde WASM memory
function getString(ptr, len) {
    const view = new Uint8Array(memory.buffer, ptr, len);
    return new TextDecoder().decode(view);
}

module.exports = { initWasm, INITIAL_PAGES, MAXIMUM_PAGES };
