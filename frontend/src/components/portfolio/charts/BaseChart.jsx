import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Skeleton } from '../../common/Skeleton';

export const CHART_PALETTE = [
  '#10b981', // emerald-500
  '#3b82f6', // blue-500
  '#f59e0b', // amber-500
  '#8b5cf6', // violet-500
  '#ef4444', // red-500
  '#06b6d4', // cyan-500
  '#ec4899', // pink-500
  '#eab308', // yellow-500
];

export function BaseChart({ option, height = 360, isLoading = false, className = '' }) {
  const mergedOption = useMemo(() => {
    const baseOption = {
      backgroundColor: 'transparent',
      color: CHART_PALETTE,
      textStyle: {
        fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
        color: '#a1a1aa',
      },
      grid: {
        top: 36,
        left: 56,
        right: 24,
        bottom: 48,
        containLabel: true,
        ...(option?.grid || {}),
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#18181b',
        borderColor: '#27272a',
        borderWidth: 1,
        textStyle: {
          color: '#f4f4f5',
          fontSize: 12,
        },
        ...(option?.tooltip || {}),
      },
      legend: {
        top: 0,
        textStyle: {
          color: '#d4d4d8',
          fontSize: 12,
        },
        ...(option?.legend || {}),
      },
    };

    return {
      ...baseOption,
      ...option,
      grid: baseOption.grid,
      tooltip: baseOption.tooltip,
      legend: option?.legend === false ? { show: false } : baseOption.legend,
    };
  }, [option]);

  if (isLoading) {
    return (
      <div className={`w-full ${className}`} style={{ height }}>
        <Skeleton className="w-full h-full" />
      </div>
    );
  }

  return (
    <div className={`w-full ${className}`} style={{ height }}>
      <ReactECharts
        option={mergedOption}
        style={{ height: `${height}px`, width: '100%' }}
        notMerge={true}
        lazyUpdate={true}
        opts={{ renderer: 'canvas' }}
      />
    </div>
  );
}

export default BaseChart;

