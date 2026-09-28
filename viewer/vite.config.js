import path from 'node:path';
import { fileURLToPath } from 'node:url';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

import { depthwizardData } from './server/dataRoute.js';

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react(), depthwizardData({ outputsDir: path.resolve(here, '..', 'data', 'outputs') })],
  server: { host: 'localhost', port: 5173 },
  test: { environment: 'node', include: ['tests/**/*.test.js'] },
});
