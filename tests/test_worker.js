/**
 * @file test_worker.js
 * @description Sanity test for wasm-worker.js using Node.js worker_threads.
 */

const { Worker } = require('worker_threads');
const path = require('path');
const fs = require('fs');

const workerPath = path.resolve(__dirname, '../web/wasm-worker.js');

console.log("Starting test for wasm-worker.js...");

const worker = new Worker(workerPath);

let isReady = false;

worker.on('message', (msg) => {
    if (msg.type === 'ready') {
        console.log("Worker reported ready state.");
        isReady = true;
        
        // Send compile command
        const sourceCode = `
        __global__ void test_worker(float *out, int n) {
            int tid = threadIdx.x;
            if (tid < n) out[tid] = 1.0f;
        }`;
        
        console.log("Sending compile command...");
        worker.postMessage({
            command: 'compile',
            source: sourceCode,
            args: ['/input.cu', '-o', '/out.ptx', '--nvidia-ptx']
        });
        
    } else if (msg.type === 'stdout') {
        console.log("[Worker stdout]:", msg.payload);
    } else if (msg.type === 'stderr') {
        console.log("[Worker stderr]:", msg.payload);
    } else if (msg.type === 'compile_result') {
        console.log("Worker returned compile result.");
        
        if (msg.exitCode !== 0) {
            console.error("Error: Worker compilation failed with exit code", msg.exitCode);
            if (msg.error) console.error("Error Details:", msg.error);
            cleanUpAndExit(1);
        }
        
        if (!msg.output || !msg.output.includes('.entry test_worker')) {
            console.error("Error: Worker compilation output invalid. Got:\n", msg.output);
            cleanUpAndExit(1);
        }
        
        console.log("Worker compilation and output reading passed!");
        cleanUpAndExit(0);
    }
});

worker.on('error', (err) => {
    console.error("Worker encountered an error:", err);
    cleanUpAndExit(1);
});

worker.on('exit', (code) => {
    if (code !== 0) {
        console.error("Worker stopped with exit code " + code);
        cleanUpAndExit(code);
    }
});

function cleanUpAndExit(code) {
    process.exit(code);
}
