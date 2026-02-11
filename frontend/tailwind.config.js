/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      colors: {
        surface: {
          0: "#000000",
          1: "#0a0a0a",
          2: "#111111",
          3: "#1a1a1a",
          4: "#222222",
          5: "#2a2a2a",
        },
        border: {
          DEFAULT: "#2a2a2a",
          hover: "#3a3a3a",
          active: "#4a4a4a",
        },
        accent: {
          theory: "#6b8afd",
          code: "#4ade80",
          math: "#c084fc",
          hybrid: "#94a3b8",
          error: "#f87171",
          warn: "#fbbf24",
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
