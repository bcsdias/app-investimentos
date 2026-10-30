import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Skeleton } from '../../common/Skeleton';
import { ErrorState } from '../../common/ErrorState';

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

const AXIS_NAME_GRID_TOP = 64;

export function BaseChart({
  option,
  height = 360,
  isLoading = false,
  error = null,
  onRetry,
  className = '',
}) {
  const mergedOption = useMemo(() => {
    // Nome de eixo Y fica ~15px acima do grid, na mesma faixa da legenda (top: 0, ~24px).
    // Com legenda e nome de eixo, o grid desce o bastante para os dois caberem.
    const yAxes = [].concat(option?.yAxis || []);
    const hasYAxisName = option?.legend !== false && yAxes.some((ax) => ax?.name);
    const gridTop = option?.grid?.top ?? 36;
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
        top: hasYAxisName ? Math.max(gridTop, AXIS_NAME_GRID_TOP) : gridTop,
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
        type: 'scroll',
        top: 0,
        textStyle: {
          color: '#d4d4d8',
          fontSize: 12,
        },
        pageIconColor: '#a1a1aa',
        pageIconInactiveColor: '#3f3f46',
        pageTextStyle: {
          color: '#a1a1aa',
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

  if (error) {
    return (
      <div className={`w-full ${className}`} style={{ height }}>
        <ErrorState error={error} onRetry={onRetry} className="h-full" />
      </div>
    );
  }

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

