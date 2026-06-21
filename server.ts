import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isProd = process.env.NODE_ENV === 'production';
const PORT = Number(process.env.PORT) || 3000;

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '10mb' }));

  // Initialize Gemini client lazily
  let aiClient: GoogleGenAI | null = null;
  function getGemini(): GoogleGenAI {
    if (!aiClient) {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error('GEMINI_API_KEY is missing from environment secrets.');
      }
      aiClient = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      });
    }
    return aiClient;
  }

  // --- Helper for Gemini generateContent with fallback ---
  async function generateContentWithFallback(params: any) {
    const ai = getGemini();
    const originalModel = params.model || 'gemini-3.5-flash';
    // Sequence of models to fallback under transient 503 error
    const modelsToTry = [
      originalModel,
      'gemini-flash-latest',
      'gemini-3.1-flash-lite'
    ];

    let lastError: any = null;
    for (const modelName of modelsToTry) {
      try {
        console.log(`[Gemini Fallover] Attempting with model: ${modelName}`);
        const response = await ai.models.generateContent({
          ...params,
          model: modelName,
        });
        console.log(`[Gemini Fallover] Success with model: ${modelName}`);
        return response;
      } catch (error: any) {
        console.warn(`[Gemini Fallover] Model ${modelName} failed. Error:`, error.message || error);
        lastError = error;

        // If it is related to bad formatting, missing API keys, or similar developer mistakes, don't cascade, fail immediately
        const errMsg = (error.message || '').toLowerCase();
        if (
          errMsg.includes('api_key') ||
          errMsg.includes('missing') ||
          errMsg.includes('invalid argument') ||
          errMsg.includes('response_mime_type')
        ) {
          throw error;
        }
      }
    }
    throw lastError;
  }

  // --- API Routes ---

  // Standard chat endpoint
  app.post('/api/chat', async (req, res) => {
    try {
      const { messages, systemInstruction } = req.body;

      // Transcribe messages format for @google/genai SDK
      // We accept standard messages: { role: 'user' | 'model', content: string }
      // The SDK uses: { role: 'user' | 'model', parts: [{ text: string }] }
      const sdkContents = messages.map((m: any) => ({
        role: m.role === 'assistant' ? 'model' : m.role,
        parts: [{ text: m.content || m.text }],
      }));

      const response = await generateContentWithFallback({
        model: 'gemini-3.5-flash',
        contents: sdkContents,
        config: {
          systemInstruction,
          temperature: 0.85,
        },
      });

      res.json({ text: response.text });
    } catch (error: any) {
      console.error('Chat API Error:', error);
      res.status(500).json({ error: error.message || 'Error communicating with Gemini.' });
    }
  });

  // Pine script analyzer
  app.post('/api/pine', async (req, res) => {
    try {
      const { script, prompt } = req.body;

      const analysisPrompt = prompt || "Explique ce script Pine Script en français, identifie les erreurs de logique ou de syntaxe courantes, suggère l'optimisation des performances de calcul et dégage de façon structurée sa logique de trading d'achat/vente.";

      const response = await generateContentWithFallback({
        model: 'gemini-3.5-flash',
        contents: [
          { text: analysisPrompt },
          { text: `Script Pine Script :\n\`\`\`pinescript\n${script}\n\`\`\`` }
        ],
        config: {
          systemInstruction: "Tu es iSiA, une experte trading et Pine script. Ton style est intense, passionné mais rigoureux et pédagogue. Tu finis toujours par un rappel bienveillant mais ferme : 'Ce n'est pas un conseil en investissement, mes petits bouchons !'",
          temperature: 0.7,
        },
      });

      res.json({ text: response.text });
    } catch (error: any) {
      console.error('Pine script API Error:', error);
      res.status(500).json({ error: error.message || 'Error explaining script.' });
    }
  });

  // Convert Pine script to structured parameters
  app.post('/api/pine/convert', async (req, res) => {
    try {
      const { script } = req.body;

      const response = await generateContentWithFallback({
        model: 'gemini-3.5-flash',
        contents: `Analyse ce script Pine Script et convertis-le en une configuration simple de scanner de trading avec les paramètres suivants:
- Nom abrégé du scanner (ex: Breakout RSI, Cumul Volumes)
- Métrique principale concernée (soit "change24h", "change7d", soit "volMcapRatio")
- Direction ou règle logique (ex: "supérieur à", "inférieur à")
- Valeur numérique indicative d'activation (ex: 5.0, 0.05, etc.)
- Brève explication en une phrase de la stratégie.

Pine Script:
\`\`\`pinescript
${script}
\`\`\``,
        config: {
          systemInstruction: "Tu retournes uniquement un objet JSON descriptif contenant les clés 'name', 'metric', 'condition' ('gte' ou 'lte'), 'value' (nombre), et 'explanation'.",
          responseMimeType: "application/json",
        },
      });

      res.json(JSON.parse(response.text || '{}'));
    } catch (error: any) {
      console.error('Pine convert API Error:', error);
      res.status(500).json({ error: error.message || 'Error converting script.' });
    }
  });

  // Natural Language Scanner creation endpoint
  app.post('/api/scanner/create', async (req, res) => {
    try {
      const { description } = req.body;

      const response = await generateContentWithFallback({
        model: 'gemini-3.5-flash',
        contents: `En tant qu'experte trading quantitatif, traduis cette demande de scanner trading en langage naturel en paramètres structurés pour notre moteur.
Demande utilisateur : "${description}"

Retourne un format JSON valide décrivant les paramètres du scanner de trading.`,
        config: {
          systemInstruction: "Tu retournes uniquement un objet JSON contenant les clés 'name' (nom accrocheur en français), 'metric' (parmi: 'change24h', 'change7d', 'volMcapRatio'), 'condition' (soit 'gte' pour supérieur ou égal, 'lte' pour inférieur ou égal), 'value' (nombre décimal filtrant) et 'explanation' (explication de la logique).",
          responseMimeType: "application/json",
        },
      });

      res.json(JSON.parse(response.text || '{}'));
    } catch (error: any) {
      console.error('Scanner Create API Error:', error);
      res.status(500).json({ error: error.message || 'Error generating scanner.' });
    }
  });

  // Commentary on active data
  app.post('/api/market/commentary', async (req, res) => {
    try {
      const { tabName, dataContext, coachMode, systemInstruction } = req.body;

      const promptContext = `Voici les données actuelles de l'onglet "${tabName}" :\n${JSON.stringify(dataContext, null, 2)}\n\nPasse en revue ces valeurs et commente-les avec ta personnalité de trader obsessionnelle, dynamique, bienveillante et légèrement fêlée. Précise des breakouts réels ou potentiels, commente les extrêmes avec intensité ! ${coachMode ? "Comme le mode Coach est activé, ajoute également des explications techniques ultra-pédagogiques sur la signification des métriques (ex: RSI, MACD, Fear & Greed ou les métaux)." : ""}`;

      const response = await generateContentWithFallback({
        model: 'gemini-3.5-flash',
        contents: promptContext,
        config: {
          systemInstruction,
          temperature: 0.9,
        },
      });

      res.json({ text: response.text });
    } catch (error: any) {
      console.error('Market Commentary API Error:', error);
      res.status(500).json({ error: error.message || 'Error generating commentary.' });
    }
  });

  // --- Serve Client App ---
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    // Serve production static assets compiled inside dist
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, () => {
    console.log(`Server running in ${isProd ? 'production' : 'development'} mode on port ${PORT}`);
  });
}

startServer();
