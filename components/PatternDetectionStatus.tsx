'use client';

import { PatternSettings } from '@/types/patternSettings';

interface PatternDetectionStatusProps {
  settings: PatternSettings;
  patternCount: number;
  isDetecting: boolean;
}

export default function PatternDetectionStatus({
  settings,
  patternCount,
  isDetecting,
}: PatternDetectionStatusProps) {
  return (
    <div
      className="border p-2 mb-2 text-xs"
      style={{
        background: 'var(--bg-3)',
        borderColor: 'var(--bg-1)',
        color: 'var(--text-3)',
      }}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="font-semibold" style={{ color: 'var(--text-2)' }}>
          Active Settings
        </span>
        {isDetecting && (
          <span className="animate-pulse" style={{ color: 'var(--accent)' }}>
            Detecting...
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
        <div>
          <span style={{ color: 'var(--text-4)' }}>Windows:</span>{' '}
          <span className="font-mono">{settings.detectionWindows.slice(0, 3).join(', ')}...</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-4)' }}>Min Pts:</span>{' '}
          <span className="font-mono">{settings.minWindow}</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-4)' }}>Max Total:</span>{' '}
          <span className="font-mono">{settings.maxPatterns}</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-4)' }}>Max/Type:</span>{' '}
          <span className="font-mono">{settings.maxPatternsPerType}</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-4)' }}>Min Conf:</span>{' '}
          <span className="font-mono">{(settings.minConfidence * 100).toFixed(0)}%</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-4)' }}>Detected:</span>{' '}
          <span className="font-mono font-semibold" style={{ color: 'var(--accent)' }}>
            {patternCount}
          </span>
        </div>
      </div>
    </div>
  );
}
