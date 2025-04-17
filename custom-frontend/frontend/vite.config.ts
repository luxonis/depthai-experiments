import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// https://vite.dev/config/
export default defineConfig({
	plugins: [react(),],
	// This is needed by FoxGlove
	define: {
		global: {},
	},
	worker: {
		format: "es",
	},
	build: {
		outDir: '../encoded-stream/frontend/dist',
		emptyOutDir: true,
		rollupOptions: {
			output: {
				format: "esm",
			},
		},
	},
});
