/**
 * @file wasm-worker.js
 * @description Web Worker script to handle asynchronous BarraCUDA compilation.
 * This runs the WebAssembly compiler off the main thread to keep the UI responsive.
 */

/**
 * Emscripten module configuration object.
 * We override print and printErr to capture standard output and standard error
 * and post them back to the main thread.
 * @type {Object}
 */
var Module = {
    /**
     * Captures standard output from the WASM module.
     * @param {string} text - The output string.
     */
    print: function(text) {
        postMessage({ type: 'stdout', payload: text });
    },
    
    /**
     * Captures standard error from the WASM module.
     * @param {string} text - The error string.
     */
    printErr: function(text) {
        postMessage({ type: 'stderr', payload: text });
    },
    
    /**
     * Called when the WASM runtime has fully initialized.
     */
    onRuntimeInitialized: function() {
        postMessage({ type: 'ready' });
    }
};

// In a browser environment, importScripts is available.
// In Node.js testing environment, it might not be, so we handle it gracefully.
if (typeof importScripts === 'function') {
    importScripts('barracuda.js');
} else if (typeof require === 'function') {
    // For Node.js Worker Thread testing
    const path = require('path');
    const { parentPort } = require('worker_threads');
    
    // Mock self and postMessage for Node.js worker_threads
    global.self = {};
    global.postMessage = (msg) => {
        if (parentPort) parentPort.postMessage(msg);
    };

    // Inject our Module into global so Emscripten might pick it up, 
    // but also we assign it explicitly. Actually, the easiest way to override
    // Emscripten in CommonJS is to just set properties on the exported module.
    const wasmPath = path.resolve(__dirname, '../web/barracuda.js');
    const emscriptenModule = require(wasmPath);
    
    // Patch the emscripten module with our handlers
    emscriptenModule.print = Module.print;
    emscriptenModule.printErr = Module.printErr;
    emscriptenModule.onRuntimeInitialized = Module.onRuntimeInitialized;
    
    // Replace our local Module variable with the fully loaded one
    Module = emscriptenModule;

    // In Node.js, we must manually trigger onRuntimeInitialized if it already initialized
    if (Module.calledRun) {
        Module.onRuntimeInitialized();
    }
    
    // Hook parentPort to self.onmessage
    if (parentPort) {
        parentPort.on('message', (msg) => {
            if (typeof self.onmessage === 'function') {
                self.onmessage({ data: msg });
            }
        });
    }
}

/**
 * Listens for messages from the main thread.
 * Expected message format:
 * {
 *   command: 'compile',
 *   source: string,
 *   args: Array<string>
 * }
 */
self.onmessage = function(e) {
    const data = e.data;
    
    if (data.command === 'compile') {
        const sourceCode = data.source || '';
        // Extract output filename from args to read it back later.
        // We look for "-o" and take the next argument.
        let outputFilename = '/out.ptx'; // default fallback
        const args = data.args || [];
        
        for (let i = 0; i < args.length - 1; i++) {
            if (args[i] === '-o') {
                outputFilename = args[i+1];
                break;
            }
        }
        
        try {
            // Write the incoming source code to a virtual file
            Module.FS.writeFile(args[0] || '/input.cu', sourceCode);
            
            // Execute the compiler
            // Expected args: e.g. ['/input.cu', '-o', '/out.ptx', '--nvidia-ptx']
            const exitCode = Module.callMain(args);
            
            let outputData = null;
            
            if (exitCode === 0) {
                // Read the generated output file from the virtual file system
                try {
                    const stat = Module.FS.stat(outputFilename);
                    if (stat && stat.size > 0) {
                        outputData = Module.FS.readFile(outputFilename, { encoding: 'utf8' });
                    } else {
                        outputData = "// Compilation succeeded, but output file is empty.";
                    }
                } catch (err) {
                    outputData = "// Compilation succeeded, but could not read output file.";
                }
            }
            
            // Send the result back to the main thread
            postMessage({
                type: 'compile_result',
                exitCode: exitCode,
                output: outputData
            });
            
        } catch (err) {
            // Catch any unexpected errors from the Emscripten runtime or FS
            postMessage({
                type: 'compile_result',
                exitCode: -1,
                output: null,
                error: err.message || err.toString()
            });
        }
    }
};
