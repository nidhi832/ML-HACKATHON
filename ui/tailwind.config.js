/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0B1120',
        panel: '#111827',
        'panel-light': '#1E293B',
        border: '#253247',
        'border-light': '#334155',
        primary: {
          DEFAULT: '#3B82F6', // Electric blue
          hover: '#2563EB',
          dark: '#1D4ED8',
        },
        success: '#10B981', // Emerald green
        warning: '#F59E0B', // Amber
        error: '#EF4444',   // Red
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      borderRadius: {
        card: '12px',
      }
    },
  },
  plugins: [],
}
