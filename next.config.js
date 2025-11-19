/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEWS_API_KEY: process.env.NEWS_API_KEY,
  },
  // Optimize performance
  swcMinify: true,
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production' ? { exclude: ['error', 'warn'] } : false,
  },
  // Configure for transformers.js server-side usage and bundle optimization
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Externalize node-specific packages for server builds
      config.externals = [...(config.externals || []), 'sharp', 'onnxruntime-node'];
    }

    // Optimize bundle splitting for large libraries
    if (!isServer) {
      config.optimization = {
        ...config.optimization,
        moduleIds: 'deterministic',
        splitChunks: {
          chunks: 'all',
          cacheGroups: {
            // Split TensorFlow.js into its own chunk
            tensorflow: {
              test: /[\\/]node_modules[\\/]@tensorflow[\\/]tfjs[\\/]/,
              name: 'tensorflow',
              priority: 30,
              reuseExistingChunk: true,
            },
            // Split WebGPU backend separately for better caching
            tensorflowWebgpu: {
              test: /[\\/]node_modules[\\/]@tensorflow[\\/]tfjs-backend-webgpu[\\/]/,
              name: 'tensorflow-webgpu',
              priority: 31,
              reuseExistingChunk: true,
            },
            // Split transformers into its own chunk
            transformers: {
              test: /[\\/]node_modules[\\/]@xenova[\\/]/,
              name: 'transformers',
              priority: 30,
              reuseExistingChunk: true,
            },
            // Split Recharts into its own chunk
            recharts: {
              test: /[\\/]node_modules[\\/]recharts[\\/]/,
              name: 'recharts',
              priority: 20,
              reuseExistingChunk: true,
            },
            // Default vendors chunk
            defaultVendors: {
              test: /[\\/]node_modules[\\/]/,
              priority: 10,
              reuseExistingChunk: true,
            },
          },
        },
      };
    }

    return config;
  },
}

module.exports = nextConfig
