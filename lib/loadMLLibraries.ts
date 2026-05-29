// Lazily import the heavy ML / pattern libraries (TensorFlow.js-backed) only
// when a forecast is actually requested, so they're not in the initial bundle.
// Extracted from app/page.tsx.

export const loadMLLibraries = async () => {
  const [
    { generateMLForecast },
    { generateProphetWithChangepoints },
    { generateLinearRegression, generateEMAForecast, generateARIMAForecast, generateProphetLiteForecast },
    { generateGRUForecast, generateCNNLSTMForecast, generateEnsembleFromPredictions },
    { detectChartPatterns },
  ] = await Promise.all([
    import('@/lib/mlForecasting'),
    import('@/lib/prophetForecast'),
    import('@/lib/mlAlgorithms'),
    import('@/lib/advancedMLModels'),
    import('@/lib/chartPatterns'),
  ]);

  return {
    generateMLForecast,
    generateProphetWithChangepoints,
    generateLinearRegression,
    generateEMAForecast,
    generateARIMAForecast,
    generateProphetLiteForecast,
    generateGRUForecast,
    generateCNNLSTMForecast,
    generateEnsembleFromPredictions,
    detectChartPatterns,
  };
};
