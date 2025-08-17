/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Ultra-dark monochrome palette
        'bg-0': '#0A0A0B',
        'bg-1': '#0E0E10',
        'bg-2': '#151517',
        'bg-3': '#1B1B1E',
        'ink-1': '#2A2A2F',
        'ink-2': '#3A3A42',
        'ink-3': '#4A4A55',
        'txt-1': '#D7D7DC',
        'txt-2': '#A5A5AC',
        'txt-3': '#7A7A82',
        'acc-1': '#5A5A66',
        'acc-2': '#70707B'
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}