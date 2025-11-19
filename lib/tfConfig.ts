/**
 * TensorFlow.js Configuration for Production Performance
 * Optimizes TF.js for browser execution with GPU acceleration
 */

import * as tf from '@tensorflow/tfjs';
// Import and register WebGPU backend
import '@tensorflow/tfjs-backend-webgpu';

let isInitialized = false;
let initializationPromise: Promise<void> | null = null;

/**
 * Initialize TensorFlow.js with WebGPU backend for optimal performance
 * WebGPU is 2-3x faster than WebGL for ML workloads
 */
export async function initializeTensorFlow(): Promise<void> {
  // If already initialized, return immediately
  if (isInitialized) {
    console.log('ℹ️  TensorFlow.js already initialized, current backend:', tf.getBackend());
    return;
  }

  // If initialization is in progress, wait for it
  if (initializationPromise) {
    console.log('⏳ Waiting for TensorFlow.js initialization to complete...');
    return initializationPromise;
  }

  // Start initialization
  initializationPromise = (async () => {
    try {
      console.log('🚀 Initializing TensorFlow.js with WebGPU backend...');
      console.log(`   Available backends: ${tf.engine().backendNames().join(', ')}`);

      // Try to set WebGPU backend first (fastest)
      try {
        const success = await tf.setBackend('webgpu');
        if (!success) {
          throw new Error('WebGPU backend registration failed');
        }
        await tf.ready();
        const actualBackend = tf.getBackend();
        console.log(`✅ WebGPU backend initialized successfully!`);
        console.log(`   Active backend: ${actualBackend}`);
        console.log(`   Memory: ${JSON.stringify(tf.memory())}`);

        if (actualBackend !== 'webgpu') {
          console.warn(`⚠️  Expected webgpu but got ${actualBackend}, trying WebGL...`);
          throw new Error(`Backend mismatch: expected webgpu, got ${actualBackend}`);
        }
      } catch (webgpuError) {
        console.warn('⚠️  WebGPU not available, falling back to WebGL');
        console.warn('   Reason:', webgpuError);

        // Fallback to WebGL (default)
        await tf.setBackend('webgl');
        await tf.ready();
        const actualBackend = tf.getBackend();
        console.log(`✅ WebGL backend initialized`);
        console.log(`   Active backend: ${actualBackend}`);
      }

    // Configure TensorFlow.js for production performance
    tf.env().set('WEBGL_VERSION', 2);
    tf.env().set('WEBGL_CPU_FORWARD', false); // Force GPU execution
    tf.env().set('WEBGL_PACK', true); // Enable texture packing for faster ops
    tf.env().set('WEBGL_FORCE_F16_TEXTURES', false); // Use F32 for accuracy
    tf.env().set('WEBGL_RENDER_FLOAT32_CAPABLE', true);
    tf.env().set('WEBGL_FLUSH_THRESHOLD', -1); // Auto flush (better for most cases)

    // Enable production mode flags
    tf.env().set('IS_BROWSER', true);
    tf.env().set('PROD', true);

    // Memory management settings
    tf.engine().startScope(); // Enable automatic memory management

      isInitialized = true;
      console.log('✅ TensorFlow.js fully configured for production');
    } catch (error) {
      console.error('❌ Failed to initialize TensorFlow.js:', error);
      throw error;
    }
  })();

  return initializationPromise;
}

/**
 * Get current backend information for debugging
 */
export function getTensorFlowInfo(): {
  backend: string;
  memory: tf.MemoryInfo;
  flags: Record<string, any>;
} {
  return {
    backend: tf.getBackend(),
    memory: tf.memory(),
    flags: {
      WEBGL_VERSION: tf.env().getNumber('WEBGL_VERSION'),
      WEBGL_CPU_FORWARD: tf.env().getBool('WEBGL_CPU_FORWARD'),
      WEBGL_PACK: tf.env().getBool('WEBGL_PACK'),
      PROD: tf.env().getBool('PROD'),
    },
  };
}

/**
 * Clean up TensorFlow.js memory
 * Call this when switching stocks or after heavy operations
 */
export function cleanupTensorFlowMemory(): void {
  const before = tf.memory();
  console.log('🧹 Cleaning up TensorFlow.js memory...');
  console.log(`   Before: ${before.numTensors} tensors, ${(before.numBytes / 1024 / 1024).toFixed(2)} MB`);

  // Dispose leaked tensors
  tf.engine().endScope();
  tf.engine().startScope();

  const after = tf.memory();
  console.log(`   After: ${after.numTensors} tensors, ${(after.numBytes / 1024 / 1024).toFixed(2)} MB`);
  console.log(`   ✅ Freed ${before.numTensors - after.numTensors} tensors, ${((before.numBytes - after.numBytes) / 1024 / 1024).toFixed(2)} MB`);
}

/**
 * Force GPU warmup - run a small operation to initialize GPU
 * This prevents the first model from being slow
 */
export async function warmupGPU(): Promise<void> {
  const x = tf.randomNormal([10, 10]);
  const y = tf.randomNormal([10, 5]);
  const result = tf.matMul(x, y);
  await result.data(); // Force execution
  x.dispose();
  y.dispose();
  result.dispose();
  console.log('🔥 GPU warmed up');
}
