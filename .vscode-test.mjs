import { defineConfig } from '@vscode/test-cli';

export default defineConfig({
	files: 'tests/out/ext/**/*.test.js',
});
