/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        serif: ['Merriweather', 'Georgia', 'serif'],
      },
      colors: {
        charcoal: {
          900: '#0a0a0a',
          800: '#141414',
          700: '#1a1a1a',
          600: '#1f1f1f',
          500: '#2a2a2a',
          400: '#3a3a3a',
          300: '#4a4a4a',
        },
        gold: {
          50: '#fdf8e8',
          100: '#f9edc3',
          200: '#f2d98a',
          300: '#e8c352',
          400: '#d4af37',
          500: '#b8962e',
          600: '#967825',
          700: '#745c1d',
          800: '#524015',
          900: '#30250d',
        },
        'primary-text': '#e8e8e8',
        'secondary-text': '#a0a0a0',
        'muted-text': '#6a6a6a',
        'inverse-text': '#0a0a0a',
      },
    },
  },
  plugins: [],
}
