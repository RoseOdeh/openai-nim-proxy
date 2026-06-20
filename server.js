const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true }));

// NVIDIA NIM config
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error("❌ NIM_API_KEY is missing!");
}

// Model mapping (اختياري فقط)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Models list
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(m => ({
      id: m,
      object: 'model',
      created: Date.now(),
      owned_by: 'nim-proxy'
    }))
  });
});

// MAIN ENDPOINT (PASS THROUGH)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel =
      MODEL_MAPPING[model] || model || 'meta/llama-3.1-8b-instruct';

    // 🔥 IMPORTANT: NO MODIFICATION AT ALL
    const nimRequest = {
      model: nimModel,
      messages,
      temperature,
      max_tokens,
      stream: stream || false
    };

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 60000
      }
    );

    // STREAM MODE
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');

      response.data.on('data', (chunk) => {
        res.write(chunk);
      });

      response.data.on('end', () => res.end());
      return;
    }

    // NORMAL RESPONSE (PASS RAW BACK)
    res.json(response.data);

  } catch (error) {
    console.error("🔥 ERROR:", error.message);

    res.status(500).json({
      error: {
        message: error.message,
        details: error.response?.data || null
      }
    });
  }
});

// 404
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: "Not found" } });
});

// START SERVER
app.listen(PORT, () => {
  console.log(`🚀 Proxy running on port ${PORT}`);
});
