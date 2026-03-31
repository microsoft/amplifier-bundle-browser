# amplifier-bundle-browser

Browser runtime for Amplifier - run AI agents entirely in web browsers.

## Overview

This bundle enables running Amplifier applications in web browsers using:

- **Pyodide** - Python interpreter compiled to WebAssembly
- **amplifier-core** - The Amplifier kernel running in Pyodide
- **AmplifierBrowser** - JS facade that handles all complexity internally

## Quick Start

### Try It Now

Ask an Amplifier session with this bundle:

```
Build me a WebLLM chat application as a single HTML file. 
It should have a nice dark theme UI with a loading progress bar 
while the model loads, and support streaming responses. 
Use the Phi-3.5-mini model.
```

**What happens:**
1. The `browser:browser-developer` agent handles the request
2. It builds `amplifier-browser.js` (if not already built)
3. Creates a complete HTML file using `AmplifierBrowser`
4. The result works when served via HTTP (WebGPU requires localhost or HTTPS)

**Success criteria:**
- Uses `amplifier-browser.js` with embedded wheels
- Uses `new AmplifierBrowser({...})` pattern
- Does NOT try to install from PyPI (amplifier-core isn't there!)
- Shows loading progress while model downloads
- Chat works with streaming responses

### Manual Setup

**Step 1: Build the JS bundle**

```bash
cd amplifier-bundle-browser
python scripts/build-bundle.py \
    --core-wheel /path/to/amplifier_core-X.X.X-py3-none-any.whl \
    --foundation-wheel /path/to/amplifier_foundation-X.X.X-py3-none-any.whl
# Output: dist/amplifier-browser.js (270KB with embedded assets)
```

**Step 2: Use in your HTML**

```html
<!-- Load Pyodide -->
<script src="https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js"></script>

<!-- Load the built bundle -->
<script src="amplifier-browser.js"></script>

<script type="module">
  import { CreateMLCEngine } from 'https://esm.run/@mlc-ai/web-llm';
  window.CreateMLCEngine = CreateMLCEngine;
  
  const amp = new AmplifierBrowser({
    model: 'Phi-3.5-mini-instruct-q4f16_1-MLC',
    systemPrompt: 'You are a helpful assistant.',
    onProgress: (stage, pct, msg) => console.log(msg),
    onError: (err) => console.error(err)
  });
  
  await amp.init();
  
  // Non-streaming
  const result = await amp.execute('Hello!');
  
  // Streaming
  await amp.executeStreaming('Tell me a story', (chunk) => {
    if (chunk.type === 'delta') console.log(chunk.delta);
  });
</script>
```

### Critical: What NOT To Do

```javascript
// ❌ WRONG - amplifier-core is NOT on PyPI, this ALWAYS fails
await micropip.install('amplifier-core');

// ❌ WRONG - Manual wheel handling is error-prone
// Let AmplifierBrowser handle it

// ✅ CORRECT - Use the AmplifierBrowser facade
const amp = new AmplifierBrowser({ model: '...' });
await amp.init();  // Handles wheels, bridges, session internally
```

See [`templates/quickstart.html`](templates/quickstart.html) for a complete working example.

## What's Included

### Tools

| Tool | Description |
|------|-------------|
| `tool-browser-storage` | localStorage/IndexedDB access |
| `tool-todo` | Task tracking (in-memory) |

### Agents

| Agent | Description |
|-------|-------------|
| `browser-developer` | Expert for building browser Amplifier integrations |

### Context

- `browser-guide.md` - Browser capabilities, constraints, and patterns

## Directory Structure

```
amplifier-bundle-browser/
├── bundle.yaml                 # Main entry point
├── behaviors/
│   └── browser-runtime.yaml    # Core browser behavior
├── agents/
│   └── browser-developer.md    # Integration expert
├── context/
│   └── browser-guide.md        # Browser-specific guidance
└── modules/
    └── tool-browser-storage/   # Browser storage tool
```

## Browser Requirements

| Browser | Support |
|---------|---------|
| Chrome 113+ | ✅ Full (WebGPU for local inference) |
| Edge 113+ | ✅ Full |
| Safari 18+ | ✅ Full |
| Firefox | ⚠️ WebGPU behind flag |

## Constraints vs Desktop Amplifier

| Feature | Desktop | Browser |
|---------|---------|---------|
| Filesystem access | ✅ Full | ❌ None |
| Shell commands | ✅ Full | ❌ None |
| Persistent storage | ✅ Files | ⚠️ localStorage/IndexedDB |
| Memory | ✅ System RAM | ⚠️ Tab limit (~2-4GB) |
| Network | ✅ Full | ⚠️ CORS restrictions |

## Composition Examples

### With WebLLM (Local Inference)

```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-browser@main
  - bundle: git+https://github.com/microsoft/amplifier-bundle-webllm@main
```

### With OpenAI (Cloud API)

```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-browser@main

providers:
  - module: provider-openai
    source: git+https://github.com/microsoft/amplifier-module-provider-openai@main
    config:
      api_key: ${OPENAI_API_KEY}  # User provides at runtime
```

## Development

### Using the browser-developer Agent

The `browser-developer` agent can help you build browser Amplifier integrations:

```
"Help me create a browser chat interface with WebLLM"
"Debug why my browser Amplifier app isn't loading"
"Show me how to add persistent storage to my browser app"
```

### Testing Locally

```bash
# Serve the examples directory
cd examples
python -m http.server 8080

# Open http://localhost:8080 in browser
```

## License

MIT

## Related

- [amplifier-bundle-webllm](https://github.com/microsoft/amplifier-bundle-webllm) - WebLLM provider
- [amplifier-core](https://github.com/microsoft/amplifier-core) - Amplifier kernel
- [Pyodide](https://pyodide.org/) - Python in WebAssembly
- [WebLLM](https://webllm.mlc.ai/) - Browser-based LLM inference

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
