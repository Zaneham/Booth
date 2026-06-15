/**
 * test_wasm_run.js
 * 
 * Sanity test to verify the BarraCUDA WebAssembly build works in Node.js.
 * This loads the compiled module, waits for the runtime to initialize,
 * writes a dummy input file via Emscripten's FS API, calls the compiler main function,
 * and reads the output file to ensure everything works correctly.
 */

const Module = require('../web/barracuda.js');

Module.onRuntimeInitialized = () => {
    console.log("WASM Runtime initialized. Running tests...");

    // Test 1: Run with --version
    try {
        console.log("Testing callMain(['--version'])...");
        // callMain returns the exit code
        const versionRet = Module.callMain(['--version']);
        if (versionRet !== 0) {
            console.error("Error: Expected exit code 0 for --version, got", versionRet);
            process.exit(1);
        }
        console.log("Version check passed.");
    } catch (e) {
        console.error("Exception during version check:", e);
        process.exit(1);
    }

    // Test 2: File I/O via Virtual FS
    try {
        console.log("Testing File I/O...");
        
        // Write a dummy CUDA file into the MEMFS
        const sourceCode = `
        __global__ void vector_add(float *out, float *a, float *b, int n) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < n) out[tid] = a[tid] + b[tid];
        }`;
        
        Module.FS.writeFile('/test_input.cu', sourceCode);
        
        // Compile the dummy file (assuming default flags or minimal flags)
        // We output to /test_output.ptx
        const compileRet = Module.callMain(['/test_input.cu', '-o', '/test_output.ptx', '--nvidia-ptx']);
        
        if (compileRet !== 0) {
            console.error("Error: Compiler failed on test_input.cu, exit code:", compileRet);
            process.exit(1);
        }

        // Verify output file exists and has content
        const outStat = Module.FS.stat('/test_output.ptx');
        if (outStat.size === 0) {
            console.error("Error: Output file is empty.");
            process.exit(1);
        }
        
        const outputCode = Module.FS.readFile('/test_output.ptx', { encoding: 'utf8' });
        if (!outputCode.includes('.entry vector_add')) {
            console.error("Error: Output PTX does not contain expected 'vector_add' entry. Got:\n", outputCode);
            process.exit(1);
        }
        
        console.log("File I/O test passed.");
        console.log("All WASM run tests passed successfully!");
        process.exit(0);

    } catch (e) {
        console.error("Exception during file I/O test:", e);
        process.exit(1);
    }
};
