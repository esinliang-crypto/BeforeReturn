import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#171717",
        muted: "#6b6b6b",
        line: "#e7e2dc",
        paper: "#fbfaf8",
        wash: "#f5f2ee",
        teal: "#0f766e",
        rose: "#be3455"
      },
      boxShadow: {
        soft: "0 8px 24px rgba(23, 23, 23, 0.06)"
      }
    }
  },
  plugins: []
};

export default config;

