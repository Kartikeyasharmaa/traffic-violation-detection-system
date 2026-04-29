import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/stats": "http://127.0.0.1:8000",
      "/violations": "http://127.0.0.1:8000",
      "/detectors": "http://127.0.0.1:8000",
      "/images": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
