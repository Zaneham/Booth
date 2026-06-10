/**
 * @file app.js
 * @description Main application logic for the BarraCUDA Web Compiler UI.
 * Handles DOM interactions, Monaco editor initialization, and Web Worker communication.
 */

// Pre-defined code snippets for the examples dropdown
const EXAMPLES = {
    'vector_add': {
        name: 'CUDA Vector Add',
        language: 'cpp',
        code: `__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}`
    },
    'matmul': {
        name: 'Triton Matmul (Stub)',
        language: 'python',
        code: `import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Simplified matmul stub for testing parser/lexer
    pass
`
    }
};

// DOM Elements
const editorContainer = document.getElementById('editor-container');
const exampleSelect = document.getElementById('example-select');
const targetSelect = document.getElementById('target-select');
const compileBtn = document.getElementById('compile-btn');
const outputView = document.getElementById('output-view');
const consoleView = document.getElementById('console-view');

// State
let editor = null;
let compilerWorker = null;
let isWorkerReady = false;

/**
 * Initializes the Monaco Editor.
 */
function initEditor() {
    require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.41.0/min/vs' }});
    require(['vs/editor/editor.main'], function() {
        editor = monaco.editor.create(editorContainer, {
            value: EXAMPLES['vector_add'].code,
            language: 'cpp',
            theme: 'vs-dark',
            automaticLayout: true,
            minimap: { enabled: false }
        });
    });
}

/**
 * Populates the Examples dropdown menu.
 */
function initDropdown() {
    for (const [key, ex] of Object.entries(EXAMPLES)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = ex.name;
        exampleSelect.appendChild(option);
    }

    exampleSelect.addEventListener('change', (e) => {
        const selected = EXAMPLES[e.target.value];
        if (editor) {
            editor.setValue(selected.code);
            monaco.editor.setModelLanguage(editor.getModel(), selected.language);
        }
    });
}

/**
 * Initializes the Web Worker for background compilation.
 */
function initWorker() {
    // Determine worker script path
    compilerWorker = new Worker('wasm-worker.js');
    
    compilerWorker.onmessage = function(e) {
        const msg = e.data;
        switch (msg.type) {
            case 'ready':
                isWorkerReady = true;
                compileBtn.disabled = false;
                compileBtn.textContent = 'Compile';
                appendConsoleLog("Compiler runtime ready.\n");
                break;
            case 'stdout':
                appendConsoleLog(msg.payload + "\n");
                break;
            case 'stderr':
                appendConsoleLog("ERROR: " + msg.payload + "\n");
                break;
            case 'compile_result':
                handleCompileResult(msg);
                break;
        }
    };
    
    compilerWorker.onerror = function(err) {
        appendConsoleLog("Worker error: " + err.message + "\n");
        resetCompileButton();
    };
}

/**
 * Appends text to the console view and scrolls to the bottom.
 * @param {string} text - The text to append.
 */
function appendConsoleLog(text) {
    consoleView.value += text;
    consoleView.scrollTop = consoleView.scrollHeight;
}

/**
 * Handles the completion of the compilation process.
 * @param {Object} msg - The result message from the worker.
 */
function handleCompileResult(msg) {
    resetCompileButton();
    
    if (msg.exitCode === 0) {
        appendConsoleLog("\\nCompilation succeeded.\\n");
        outputView.value = msg.output || "// No output generated.";
    } else {
        appendConsoleLog("\\nCompilation failed with exit code " + msg.exitCode + ".\\n");
        outputView.value = msg.error ? msg.error : "// Compilation failed. See console for details.";
    }
}

/**
 * Resets the compile button state.
 */
function resetCompileButton() {
    compileBtn.disabled = false;
    compileBtn.textContent = 'Compile';
}

/**
 * Triggers the compilation process by sending a message to the Web Worker.
 */
function doCompile() {
    if (!isWorkerReady || !compilerWorker || !editor) return;
    
    // Clear previous output
    outputView.value = '';
    appendConsoleLog("\n--- Starting Compilation ---\n");
    
    compileBtn.disabled = true;
    compileBtn.textContent = 'Compiling...';
    
    const sourceCode = editor.getValue();
    const target = targetSelect.value;
    
    // Determine input file extension based on language
    const currentLang = exampleSelect.value === 'matmul' ? '.py' : '.cu';
    const inputFilename = '/input' + currentLang;
    
    // Determine output file extension
    let outExt = '.ptx';
    if (target === '--amdgpu') outExt = '.s';
    if (target === '--cpu') outExt = '.o';
    if (target === '--tensix') outExt = '.elf'; // just a guess
    
    const outputFilename = '/out' + outExt;
    
    const args = [inputFilename, '-o', outputFilename, target];
    
    // Add --triton flag if it's a python file
    if (currentLang === '.py') {
        args.push('--triton');
    }
    
    compilerWorker.postMessage({
        command: 'compile',
        source: sourceCode,
        args: args
    });
}

// Bootstrap
window.addEventListener('DOMContentLoaded', () => {
    compileBtn.disabled = true;
    compileBtn.textContent = 'Loading...';
    
    initDropdown();
    initEditor();
    initWorker();
    
    compileBtn.addEventListener('click', doCompile);
});
