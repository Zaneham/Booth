/**
 * @file test_app.js
 * @description Lightweight DOM mock test for app.js to ensure event bindings and state updates work.
 */

const fs = require('fs');
const path = require('path');

// Basic DOM Mock
class DOMNode {
    constructor(tagName) {
        this.tagName = tagName;
        this.children = [];
        this.attributes = {};
        this.events = {};
        this.value = '';
        this.textContent = '';
        this.disabled = false;
        this.scrollTop = 0;
        this.scrollHeight = 100;
    }
    
    appendChild(child) {
        this.children.push(child);
    }
    
    addEventListener(event, callback) {
        if (!this.events[event]) this.events[event] = [];
        this.events[event].push(callback);
    }
    
    dispatchEvent(eventObj) {
        const cbs = this.events[eventObj.type] || [];
        for (const cb of cbs) cb(eventObj);
    }
}

const mockDocument = {
    elements: {},
    getElementById(id) {
        if (!this.elements[id]) {
            this.elements[id] = new DOMNode('div');
        }
        return this.elements[id];
    },
    createElement(tagName) {
        return new DOMNode(tagName);
    }
};

const mockWindow = new DOMNode('window');

// Mock Monaco
const mockMonaco = {
    editor: {
        create: () => ({
            getValue: () => "mock code",
            setValue: () => {},
            getModel: () => ({})
        }),
        setModelLanguage: () => {}
    }
};

global.document = mockDocument;
global.window = mockWindow;
global.monaco = mockMonaco;

// Create a global require function for the eval context
const mockRequire = (deps, cb) => {
    if (cb) cb();
};
mockRequire.config = () => {};
global.require = mockRequire;

// Mock Worker
class MockWorker {
    constructor(script) {
        this.script = script;
    }
    postMessage(msg) {
        // Simulate immediate response
        if (msg.command === 'compile') {
            setTimeout(() => {
                this.onmessage({ data: { type: 'compile_result', exitCode: 0, output: 'mock output' } });
            }, 10);
        }
    }
}
global.Worker = MockWorker;

// Load app.js
const appJsPath = path.resolve(__dirname, '../web/app.js');
const appJsCode = fs.readFileSync(appJsPath, 'utf8');

// Evaluate app.js
let testCode = appJsCode.replace(/require/g, 'mockRequire');
testCode = testCode.replace(/let /g, 'var ').replace(/const /g, 'var ');
eval(testCode);

// Trigger DOMContentLoaded
mockWindow.dispatchEvent({ type: 'DOMContentLoaded' });

// Test initial state
const btn = mockDocument.getElementById('compile-btn');
if (btn.textContent !== 'Loading...') {
    console.error("Test failed: Button should be in Loading state initially.");
    process.exit(1);
}

// Simulate Worker Ready
eval("compilerWorker.onmessage({ data: { type: 'ready' } })");

if (btn.disabled !== false || btn.textContent !== 'Compile') {
    console.error("Test failed: Button should be enabled and say 'Compile' after worker ready.");
    process.exit(1);
}

// Trigger compile
btn.dispatchEvent({ type: 'click' });

if (btn.disabled !== true || btn.textContent !== 'Compiling...') {
    console.error("Test failed: Button should be disabled and say 'Compiling...' during compile.");
    process.exit(1);
}

// Wait for mock compile result
setTimeout(() => {
    if (btn.disabled !== false || btn.textContent !== 'Compile') {
        console.error("Test failed: Button should be reset after compile.");
        process.exit(1);
    }
    
    const outputView = mockDocument.getElementById('output-view');
    if (outputView.value !== 'mock output') {
        console.error("Test failed: Output view did not receive mock output.");
        process.exit(1);
    }
    
    console.log("All app.js mock tests passed.");
    process.exit(0);
}, 50);
