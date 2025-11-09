/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEWS_API_KEY: process.env.NEWS_API_KEY,
  },
  // Configure for transformers.js server-side usage
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Externalize node-specific packages for server builds
      config.externals = [...(config.externals || []), 'sharp', 'onnxruntime-node'];
    }
    return config;
  },
}

module.exports = nextConfig
