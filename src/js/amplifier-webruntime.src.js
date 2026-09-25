/**
 * AmplifierWeb - Browser-based Amplifier runtime
 * 
 * This facade handles all Pyodide/wheel complexity internally, providing a clean API
 * for browser-based Amplifier applications.
 * 
 * @example
 * const amp = new AmplifierWeb({
 *   model: 'Phi-3.5-mini-instruct-q4f16_1-MLC',
 *   onProgress: (stage, pct, msg) => console.log(msg)
 * });
 * await amp.init();
 * const result = await amp.execute('Hello!');
 */

// =============================================================================
// EMBEDDED ASSETS (replaced by build script)
// =============================================================================

const EMBEDDED_ASSETS = {
  // These are replaced by the build script with actual base64 content
  AMPLIFIER_CORE_WHL: '%%AMPLIFIER_CORE_WHL%%',
  AMPLIFIER_CORE_WHL_FILENAME: '%%AMPLIFIER_CORE_WHL_FILENAME%%',
  AMPLIFIER_FOUNDATION_WHL: '%%AMPLIFIER_FOUNDATION_WHL%%',
  AMPLIFIER_FOUNDATION_WHL_FILENAME: '%%AMPLIFIER_FOUNDATION_WHL_FILENAME%%',
  AMPLIFIER_WEBRUNTIME_PY: '%%AMPLIFIER_WEBRUNTIME_PY%%',
};

// =============================================================================
// ERROR TYPES
// =============================================================================

class AmplifierError extends Error {
  constructor(code, message, options = {}) {
    super(message);
    this.name = 'AmplifierError';
    this.code = code;
    this.cause = options.cause;
    this.recoverable = options.recoverable ?? false;
    this.suggestion = options.suggestion;
  }
}

const ErrorCodes = {
  WEBGPU_NOT_SUPPORTED: 'WEBGPU_NOT_SUPPORTED',
  PYODIDE_LOAD_FAILED: 'PYODIDE_LOAD_FAILED',
  WHEEL_INSTALL_FAILED: 'WHEEL_INSTALL_FAILED',
  MODEL_LOAD_FAILED: 'MODEL_LOAD_FAILED',
  SESSION_CREATE_FAILED: 'SESSION_CREATE_FAILED',
  EXECUTION_FAILED: 'EXECUTION_FAILED',
  BUNDLE_LOAD_FAILED: 'BUNDLE_LOAD_FAILED',
  NOT_INITIALIZED: 'NOT_INITIALIZED',
  ALREADY_INITIALIZED: 'ALREADY_INITIALIZED',
  NETWORK_ERROR: 'NETWORK_ERROR',
  OUT_OF_MEMORY: 'OUT_OF_MEMORY',
};

// =============================================================================
// AMPLIFIER BROWSER CLASS
// =============================================================================

class AmplifierWeb {
  /**
   * Create a new AmplifierWeb instance.
   * 
   * @param {Object} config - Configuration options
   * @param {string} [config.model] - WebLLM model ID (e.g., 'Phi-3.5-mini-instruct-q4f16_1-MLC')
   * @param {string} [config.systemPrompt] - System prompt for the assistant
   * @param {string} [config.bundle] - Path to bundle YAML file
   * @param {string} [config.bundleContent] - Inline bundle YAML content
   * @param {Function} [config.onProgress] - Progress callback (stage, percent, message)
   * @param {Function} [config.onError] - Error callback
   * @param {boolean} [config.debug] - Enable debug logging
   * @param {string} [config.pyodideUrl] - Custom Pyodide CDN URL
   * @param {boolean} [config.skipWebLLM] - Skip WebLLM initialization (for API providers)
   */
  constructor(config = {}) {
    this._config = {
      model: config.model || 'Phi-3.5-mini-instruct-q4f16_1-MLC',
      systemPrompt: config.systemPrompt || 'You are a helpful AI assistant.',
      bundle: config.bundle || null,
      bundleContent: config.bundleContent || null,
      onProgress: config.onProgress || (() => {}),
      onError: config.onError || ((e) => console.error(e)),
      debug: config.debug || false,
      pyodideUrl: config.pyodideUrl || 'https://cdn.jsdelivr.net/pyodide/v0.27.0/full/',
      skipWebLLM: config.skipWebLLM || false,
    };
    
    this._pyodide = null;
    this._webllmEngine = null;
    this._isReady = false;
    this._isExecuting = false;
    this._sessionId = null;
    this._cancelRequested = false;
  }
  
  // ===========================================================================
  // PUBLIC PROPERTIES
  // ===========================================================================
  
  get isReady() { return this._isReady; }
  get isExecuting() { return this._isExecuting; }
  get sessionId() { return this._sessionId; }
  
  // ===========================================================================
  // LIFECYCLE METHODS
  // ===========================================================================
  
  /**
   * Initialize the Amplifier runtime.
   * Loads Pyodide, installs wheels, initializes WebLLM, creates session.
   */
  async init() {
    if (this._isReady) {
      throw new AmplifierError(
        ErrorCodes.ALREADY_INITIALIZED,
        'AmplifierWeb is already initialized',
        { recoverable: false }
      );
    }
    
    try {
      // 1. Check WebGPU support
      this._progress('pyodide', 0, 'Checking requirements...');
      if (!this._config.skipWebLLM && !navigator.gpu) {
        throw new AmplifierError(
          ErrorCodes.WEBGPU_NOT_SUPPORTED,
          'WebGPU is not supported in this browser',
          { 
            recoverable: false,
            suggestion: 'Use Chrome 113+, Edge 113+, or Safari 18+ for WebGPU support'
          }
        );
      }
      
      // 2. Load Pyodide
      this._progress('pyodide', 10, 'Loading Python runtime...');
      await this._loadPyodide();
      this._progress('pyodide', 100, 'Python runtime loaded');
      
      // 3. Install packages
      this._progress('packages', 0, 'Loading package manager...');
      await this._pyodide.loadPackage('micropip');
      this._progress('packages', 50, 'Installing dependencies...');
      await this._pyodide.runPythonAsync(`
        import micropip
        await micropip.install(['pydantic', 'typing-extensions'])
      `);
      this._progress('packages', 100, 'Dependencies installed');
      
      // 4. Install wheels
      this._progress('wheels', 0, 'Installing Amplifier...');
      await this._installWheels();
      this._progress('wheels', 100, 'Amplifier installed');
      
      // 5. Register JS bridges on globalThis BEFORE loading Python module
      this._progress('session', 0, 'Setting up bridges...');
      this._registerBridges();
      
      // 6. Load amplifier_browser.py
      this._progress('session', 20, 'Loading browser runtime...');
      await this._loadBrowserModule();
      
      // 7. Load WebLLM model (if not skipped)
      if (!this._config.skipWebLLM) {
        this._progress('webllm', 0, 'Loading AI model...');
        await this._loadWebLLM();
        this._progress('webllm', 100, 'Model loaded');
      }
      
      // 8. Load bundle (if specified)
      if (this._config.bundle || this._config.bundleContent) {
        this._progress('session', 40, 'Loading bundle...');
        await this._loadBundle();
      }
      
      // 9. Create Python session
      this._progress('session', 60, 'Creating session...');
      await this._createSession();
      this._progress('session', 100, 'Session created');
      
      // 10. Mark ready
      this._isReady = true;
      this._progress('ready', 100, 'Ready!');
      
    } catch (error) {
      if (error instanceof AmplifierError) {
        this._config.onError(error);
        throw error;
      }
      
      const wrapped = new AmplifierError(
        ErrorCodes.SESSION_CREATE_FAILED,
        `Initialization failed: ${error.message}`,
        { cause: error, recoverable: true }
      );
      this._config.onError(wrapped);
      throw wrapped;
    }
  }
  
  /**
   * Dispose of resources.
   */
  async dispose() {
    if (this._webllmEngine) {
      // WebLLM doesn't have a dispose method, but we can clear the reference
      this._webllmEngine = null;
    }
    this._pyodide = null;
    this._isReady = false;
    this._sessionId = null;
  }
  
  // ===========================================================================
  // EXECUTION METHODS
  // ===========================================================================
  
  /**
   * Execute a prompt and get the complete response.
   * 
   * @param {string} prompt - User message to send
   * @returns {Promise<{content: string, toolCalls?: Array, usage?: Object}>}
   */
  async execute(prompt) {
    this._ensureReady();
    this._isExecuting = true;
    this._cancelRequested = false;
    
    try {
      const resultJson = await this._pyodide.runPythonAsync(`
        import json
        result = await session.execute(${JSON.stringify(prompt)})
        json.dumps({
          'content': result,
          'toolCalls': [],
          'usage': {}
        })
      `);
      
      return JSON.parse(resultJson);
    } catch (error) {
      throw new AmplifierError(
        ErrorCodes.EXECUTION_FAILED,
        `Execution failed: ${error.message}`,
        { cause: error, recoverable: true }
      );
    } finally {
      this._isExecuting = false;
    }
  }
  
  /**
   * Execute a prompt with streaming response.
   * 
   * @param {string} prompt - User message to send
   * @param {Function} onChunk - Callback for each chunk
   * @returns {Promise<{content: string, toolCalls?: Array, usage?: Object}>}
   */
  async executeStreaming(prompt, onChunk) {
    this._ensureReady();
    this._isExecuting = true;
    this._cancelRequested = false;
    
    // Register the chunk callback
    globalThis._amplifierOnChunk = (chunkJson) => {
      try {
        const chunk = JSON.parse(chunkJson);
        onChunk(chunk);
      } catch (e) {
        this._debug('Chunk parse error:', e);
      }
    };
    
    try {
      // Use execute() since BrowserAmplifierSession doesn't have execute_streaming
      // Simulate streaming by sending content as a single chunk after completion
      const resultJson = await this._pyodide.runPythonAsync(`
        import json
        from js import _amplifierOnChunk
        
        result = await session.execute(${JSON.stringify(prompt)})
        
        # Send the complete result as a "delta" chunk
        _amplifierOnChunk(json.dumps({'type': 'delta', 'delta': result}))
        
        json.dumps({
          'content': result,
          'toolCalls': [],
          'usage': {}
        })
      `);
      
      return JSON.parse(resultJson);
    } catch (error) {
      throw new AmplifierError(
        ErrorCodes.EXECUTION_FAILED,
        `Streaming execution failed: ${error.message}`,
        { cause: error, recoverable: true }
      );
    } finally {
      this._isExecuting = false;
      delete globalThis._amplifierOnChunk;
    }
  }
  
  /**
   * Cancel the currently executing prompt.
   */
  cancel() {
    this._cancelRequested = true;
  }
  
  // ===========================================================================
  // SESSION MANAGEMENT
  // ===========================================================================
  
  /**
   * Set the system prompt.
   */
  setSystemPrompt(prompt) {
    this._config.systemPrompt = prompt;
    if (this._isReady) {
      this._pyodide.runPython(`session.set_system_prompt(${JSON.stringify(prompt)})`);
    }
  }
  
  /**
   * Get the current system prompt.
   */
  getSystemPrompt() {
    return this._config.systemPrompt;
  }
  
  /**
   * Get conversation history.
   */
  async getHistory() {
    this._ensureReady();
    const historyJson = await this._pyodide.runPythonAsync(`
      import json
      messages = session.get_history()
      json.dumps([{'role': m.role, 'content': m.content} for m in messages])
    `);
    return JSON.parse(historyJson);
  }
  
  /**
   * Clear conversation history.
   */
  async clearHistory() {
    this._ensureReady();
    await this._pyodide.runPythonAsync(`session.clear_history()`);
  }
  
  // ===========================================================================
  // TOOL MANAGEMENT
  // ===========================================================================
  
  /**
   * Register a custom tool using Python code.
   * 
   * @param {string} pythonCode - Python code defining a Tool class
   */
  async registerTool(pythonCode) {
    this._ensureReady();
    try {
      await this._pyodide.runPythonAsync(pythonCode);
    } catch (error) {
      throw new AmplifierError(
        ErrorCodes.EXECUTION_FAILED,
        `Failed to register tool: ${error.message}`,
        { cause: error, recoverable: true }
      );
    }
  }
  
  /**
   * Get list of available tool names.
   */
  async getToolNames() {
    this._ensureReady();
    const toolsJson = await this._pyodide.runPythonAsync(`
      import json
      json.dumps(session.get_tool_names())
    `);
    return JSON.parse(toolsJson);
  }
  
  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================
  
  _ensureReady() {
    if (!this._isReady) {
      throw new AmplifierError(
        ErrorCodes.NOT_INITIALIZED,
        'AmplifierWeb is not initialized. Call init() first.',
        { recoverable: true, suggestion: 'Call init() before using other methods' }
      );
    }
  }
  
  _progress(stage, percent, message) {
    this._debug(`[${stage}] ${percent}% - ${message}`);
    this._config.onProgress(stage, percent, message);
  }
  
  _debug(...args) {
    if (this._config.debug) {
      console.log('[AmplifierWeb]', ...args);
    }
  }
  
  async _loadPyodide() {
    try {
      // Check if loadPyodide is available (script must be loaded)
      if (typeof loadPyodide === 'undefined') {
        // Try to load it dynamically
        await this._loadScript(this._config.pyodideUrl + 'pyodide.js');
      }
      
      this._pyodide = await loadPyodide({
        indexURL: this._config.pyodideUrl,
      });
    } catch (error) {
      throw new AmplifierError(
        ErrorCodes.PYODIDE_LOAD_FAILED,
        `Failed to load Pyodide: ${error.message}`,
        { cause: error, recoverable: true, suggestion: 'Check network connection and retry' }
      );
    }
  }
  
  async _loadScript(url) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = url;
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Failed to load script: ${url}`));
      document.head.appendChild(script);
    });
  }
  
  async _installWheels() {
    try {
      // Decode and write amplifier-core wheel
      const coreWhlBytes = this._base64ToBytes(EMBEDDED_ASSETS.AMPLIFIER_CORE_WHL);
      const coreWhlPath = `/tmp/${EMBEDDED_ASSETS.AMPLIFIER_CORE_WHL_FILENAME}`;
      this._pyodide.FS.writeFile(coreWhlPath, coreWhlBytes);
      
      // Decode and write amplifier-foundation wheel
      const foundationWhlBytes = this._base64ToBytes(EMBEDDED_ASSETS.AMPLIFIER_FOUNDATION_WHL);
      const foundationWhlPath = `/tmp/${EMBEDDED_ASSETS.AMPLIFIER_FOUNDATION_WHL_FILENAME}`;
      this._pyodide.FS.writeFile(foundationWhlPath, foundationWhlBytes);
      
      // Install wheels with deps=False to avoid PyYAML conflict
      // MUST use Python to pass kwargs, not JS proxy
      await this._pyodide.runPythonAsync(`
        import micropip
        await micropip.install('emfs:${coreWhlPath}', deps=False)
        await micropip.install('emfs:${foundationWhlPath}', deps=False)
      `);
      
    } catch (error) {
      throw new AmplifierError(
        ErrorCodes.WHEEL_INSTALL_FAILED,
        `Failed to install wheels: ${error.message}`,
        { cause: error, recoverable: true }
      );
    }
  }
  
  _registerBridges() {
    // Store reference to this for closures
    const self = this;
    
    // MUST use globalThis, NOT pyodide.globals.set()
    // Python's `from js import X` looks in globalThis
    
    globalThis.js_llm_complete = async (messagesJson, toolsJson) => {
      const messages = JSON.parse(messagesJson);
      
      if (!self._webllmEngine) {
        throw new Error('WebLLM engine not initialized');
      }
      
      const response = await self._webllmEngine.chat.completions.create({
        messages: messages,
        temperature: 0.7,
        max_tokens: 2048,
      });
      
      return JSON.stringify({
        content: response.choices[0].message.content,
        finish_reason: response.choices[0].finish_reason,
        usage: {
          prompt_tokens: response.usage?.prompt_tokens || 0,
          completion_tokens: response.usage?.completion_tokens || 0,
          total_tokens: response.usage?.total_tokens || 0,
        }
      });
    };
    
    globalThis.js_llm_stream = async (messagesJson, onChunk) => {
      const messages = JSON.parse(messagesJson);
      const chunks = [];
      
      if (!self._webllmEngine) {
        throw new Error('WebLLM engine not initialized');
      }
      
      const stream = await self._webllmEngine.chat.completions.create({
        messages: messages,
        temperature: 0.7,
        max_tokens: 2048,
        stream: true,
      });
      
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content || '';
        if (delta) {
          chunks.push(delta);
          onChunk(JSON.stringify({ type: 'delta', content: delta }));
        }
      }
      
      return JSON.stringify({
        content: chunks.join(''),
        finish_reason: 'stop',
      });
    };
    
    globalThis.js_web_fetch = async (url) => {
      const response = await fetch(url, {
        credentials: 'omit',
        redirect: 'error',
        referrerPolicy: 'no-referrer',
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.text();
    };
  }
  
  async _loadBrowserModule() {
    try {
      const moduleCode = atob(EMBEDDED_ASSETS.AMPLIFIER_WEBRUNTIME_PY);
      // Write as a file so Python can import it
      this._pyodide.FS.writeFile('/home/pyodide/amplifier_browser.py', moduleCode);
      this._debug('Wrote amplifier_browser.py to Pyodide filesystem');
    } catch (error) {
      throw new AmplifierError(
        ErrorCodes.SESSION_CREATE_FAILED,
        `Failed to load browser module: ${error.message}`,
        { cause: error, recoverable: false }
      );
    }
  }
  
  async _loadWebLLM() {
    try {
      // Check if CreateMLCEngine is available
      if (typeof CreateMLCEngine === 'undefined') {
        // Try to load from CDN
        const { CreateMLCEngine } = await import('https://esm.run/@mlc-ai/web-llm');
        globalThis.CreateMLCEngine = CreateMLCEngine;
      }
      
      const self = this;
      this._webllmEngine = await CreateMLCEngine(this._config.model, {
        initProgressCallback: (progress) => {
          const pct = Math.round(progress.progress * 100);
          self._progress('webllm', pct, progress.text || `Loading model: ${pct}%`);
        }
      });
      
    } catch (error) {
      throw new AmplifierError(
        ErrorCodes.MODEL_LOAD_FAILED,
        `Failed to load WebLLM model: ${error.message}`,
        { 
          cause: error, 
          recoverable: true,
          suggestion: 'Try a smaller model or check GPU memory'
        }
      );
    }
  }
  
  async _loadBundle() {
    try {
      let bundleContent = this._config.bundleContent;
      
      if (!bundleContent && this._config.bundle) {
        const response = await fetch(this._config.bundle);
        if (!response.ok) {
          throw new Error(`Failed to fetch bundle: ${response.status}`);
        }
        bundleContent = await response.text();
      }
      
      // Store bundle content for Python to use
      this._bundleContent = bundleContent;
      
    } catch (error) {
      throw new AmplifierError(
        ErrorCodes.BUNDLE_LOAD_FAILED,
        `Failed to load bundle: ${error.message}`,
        { cause: error, recoverable: true }
      );
    }
  }
  
  async _createSession() {
    try {
      const modelId = JSON.stringify(this._config.model);
      const systemPrompt = JSON.stringify(this._config.systemPrompt);
      
      await this._pyodide.runPythonAsync(`
        from amplifier_browser import create_session
        
        session = create_session(
          model_id=${modelId},
          system_prompt=${systemPrompt}
        )
        await session.initialize()
      `);
      
      this._sessionId = `browser-${Date.now()}`;
      
    } catch (error) {
      throw new AmplifierError(
        ErrorCodes.SESSION_CREATE_FAILED,
        `Failed to create session: ${error.message}`,
        { cause: error, recoverable: false }
      );
    }
  }
  
  _base64ToBytes(base64) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }
}

// =============================================================================
// STATIC UTILITIES
// =============================================================================

/**
 * Check if WebGPU is supported in this browser.
 */
AmplifierWeb.isWebGPUSupported = () => {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
};

/**
 * Get the version of AmplifierWeb.
 */
AmplifierWeb.getVersion = () => {
  return '%%VERSION%%';
};

// =============================================================================
// EXPORTS
// =============================================================================

// Export for ES modules
if (typeof exports !== 'undefined') {
  exports.AmplifierWeb = AmplifierWeb;
  exports.AmplifierError = AmplifierError;
  exports.ErrorCodes = ErrorCodes;
}

// Export for browsers (global)
if (typeof window !== 'undefined') {
  window.AmplifierWeb = AmplifierWeb;
  window.AmplifierError = AmplifierError;
  window.ErrorCodes = ErrorCodes;
}
