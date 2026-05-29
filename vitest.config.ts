import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

// Tests target the pure analytical libs (indicators, signals, ML, patterns,
// correlation). Node environment is enough — no DOM needed.
export default defineConfig({
  resolve: {
    alias: {
      '@': resolve(process.cwd()),
    },
  },
  test: {
    environment: 'node',
    include: ['lib/**/__tests__/**/*.{test,spec}.ts'],
    globals: true,
  },
});
