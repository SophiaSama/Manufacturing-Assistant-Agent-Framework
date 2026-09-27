import { existsSync } from 'node:fs';
import { generateText } from 'ai';

if (existsSync('.env.local')) {
  process.loadEnvFile('.env.local');
}

const { text } = await generateText({
  model: 'openai/gpt-4o-mini',
  prompt: 'Invent a new holiday and describe its traditions.',
});

console.log(text);
