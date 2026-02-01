import { defineConfig } from 'vite'

export default defineConfig({
  // For GitHub Pages: set base to repo name
  base: '/historical-wind-direction/',
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})
