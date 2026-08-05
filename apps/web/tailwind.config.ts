import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        positive: "#16a34a",
        negative: "#dc2626",
        surface: "#0f172a",
        panel: "#1e293b",
      },
    },
  },
  plugins: [],
};

export default config;
