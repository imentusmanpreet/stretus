#!/usr/bin/env python3
"""
Analyze backtest timing logs to identify performance bottlenecks.

Usage:
    python scripts/analyze_timing.py logs/app.log
    python scripts/analyze_timing.py logs/app.log --backtest-id abc-123
    python scripts/analyze_timing.py logs/app.log --json
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_timing_log(line: str) -> dict[str, Any] | None:
    """Parse a timing log line into structured data."""
    if "⏱️  TIMING" not in line:
        return None
    
    # Extract key fields using regex
    patterns = {
        'step': r'step=([^|]+)',
        'status': r'status=([^|]+)',
        'start_time': r'start_time=([^|]+)',
        'end_time': r'end_time=([^|]+)',
        'duration': r'duration=([^|]+)',
        'duration_seconds': r'duration_seconds=([\d.]+)',
        'backtest_id': r'backtest_id=([^|]+)',
    }
    
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            value = match.group(1)
            if key == 'duration_seconds':
                result[key] = float(value)
            else:
                result[key] = value
    
    # Extract additional context
    context_match = re.search(r'\|([^|]*=.*)$', line)
    if context_match:
        context_str = context_match.group(1)
        context = {}
        for pair in context_str.split('|'):
            if '=' in pair:
                k, v = pair.split('=', 1)
                context[k] = v
        result['context'] = context
    
    return result if 'step' in result else None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def analyze_timing_logs(log_file: Path, backtest_id: str | None = None) -> dict[str, Any]:
    """Analyze timing logs and return statistics."""
    logs = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if backtest_id and backtest_id not in line:
                continue
            
            parsed = parse_timing_log(line)
            if parsed:
                logs.append(parsed)
    
    if not logs:
        return {
            'error': 'No timing logs found',
            'total_logs': 0,
        }
    
    # Separate by status
    completed = [log for log in logs if log.get('status') == 'COMPLETE']
    failed = [log for log in logs if log.get('status') == 'FAILED']
    started = [log for log in logs if log.get('status') == 'START']
    
    # Aggregate by step
    step_stats = defaultdict(lambda: {
        'count': 0,
        'total_duration': 0.0,
        'durations': [],
        'failures': 0,
    })
    
    for log in completed:
        step = log['step']
        duration = log.get('duration_seconds', 0.0)
        step_stats[step]['count'] += 1
        step_stats[step]['total_duration'] += duration
        step_stats[step]['durations'].append(duration)
    
    for log in failed:
        step = log['step']
        step_stats[step]['failures'] += 1
    
    # Calculate statistics
    step_summary = []
    for step, stats in sorted(step_stats.items()):
        if stats['durations']:
            durations = stats['durations']
            avg = stats['total_duration'] / stats['count']
            min_dur = min(durations)
            max_dur = max(durations)
            
            # Calculate percentiles
            sorted_durations = sorted(durations)
            p50_idx = len(sorted_durations) // 2
            p95_idx = int(len(sorted_durations) * 0.95)
            p50 = sorted_durations[p50_idx] if sorted_durations else 0
            p95 = sorted_durations[p95_idx] if sorted_durations else 0
            
            step_summary.append({
                'step': step,
                'count': stats['count'],
                'avg_duration': avg,
                'min_duration': min_dur,
                'max_duration': max_dur,
                'p50_duration': p50,
                'p95_duration': p95,
                'total_duration': stats['total_duration'],
                'failures': stats['failures'],
            })
    
    # Sort by total duration (descending)
    step_summary.sort(key=lambda x: x['total_duration'], reverse=True)
    
    # Find pipeline completions
    pipeline_completions = [
        log for log in completed 
        if log['step'] == 'COMPLETE_PIPELINE'
    ]
    
    return {
        'total_logs': len(logs),
        'completed': len(completed),
        'failed': len(failed),
        'started': len(started),
        'step_summary': step_summary,
        'pipeline_completions': len(pipeline_completions),
        'pipeline_durations': [
            log.get('duration_seconds', 0.0) 
            for log in pipeline_completions
        ],
    }


def print_analysis(analysis: dict[str, Any], output_json: bool = False):
    """Print analysis results."""
    if output_json:
        print(json.dumps(analysis, indent=2))
        return
    
    if 'error' in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    print("=" * 80)
    print("BACKTEST TIMING ANALYSIS")
    print("=" * 80)
    print()
    
    print(f"📊 Overview:")
    print(f"  Total timing logs: {analysis['total_logs']}")
    print(f"  Completed steps:   {analysis['completed']}")
    print(f"  Failed steps:      {analysis['failed']}")
    print(f"  Started steps:     {analysis['started']}")
    print(f"  Pipeline runs:     {analysis['pipeline_completions']}")
    print()
    
    if analysis['pipeline_durations']:
        durations = analysis['pipeline_durations']
        avg_pipeline = sum(durations) / len(durations)
        min_pipeline = min(durations)
        max_pipeline = max(durations)
        
        print(f"⏱️  Pipeline Performance:")
        print(f"  Average: {format_duration(avg_pipeline)}")
        print(f"  Min:     {format_duration(min_pipeline)}")
        print(f"  Max:     {format_duration(max_pipeline)}")
        print()
    
    print("📈 Step-by-Step Breakdown (sorted by total time):")
    print()
    print(f"{'Step':<35} {'Count':>6} {'Avg':>10} {'Min':>10} {'Max':>10} {'P95':>10} {'Total':>10} {'Fails':>6}")
    print("-" * 110)
    
    for step in analysis['step_summary']:
        print(
            f"{step['step']:<35} "
            f"{step['count']:>6} "
            f"{format_duration(step['avg_duration']):>10} "
            f"{format_duration(step['min_duration']):>10} "
            f"{format_duration(step['max_duration']):>10} "
            f"{format_duration(step['p95_duration']):>10} "
            f"{format_duration(step['total_duration']):>10} "
            f"{step['failures']:>6}"
        )
    
    print()
    print("=" * 80)
    
    # Identify bottlenecks
    print()
    print("🔍 Potential Bottlenecks:")
    print()
    
    bottlenecks = []
    for step in analysis['step_summary']:
        if step['avg_duration'] > 5.0:
            bottlenecks.append((step['step'], step['avg_duration'], 'High average duration (> 5s)'))
        elif step['max_duration'] > step['avg_duration'] * 3:
            bottlenecks.append((step['step'], step['max_duration'], 'High variance (max >> avg)'))
        elif step['failures'] > 0:
            bottlenecks.append((step['step'], step['failures'], f"{step['failures']} failures"))
    
    if bottlenecks:
        for step, value, reason in bottlenecks:
            if isinstance(value, float):
                print(f"  ⚠️  {step}: {format_duration(value)} - {reason}")
            else:
                print(f"  ⚠️  {step}: {reason}")
    else:
        print("  ✅ No significant bottlenecks detected")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze backtest timing logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all timing logs
  python scripts/analyze_timing.py logs/app.log
  
  # Analyze specific backtest
  python scripts/analyze_timing.py logs/app.log --backtest-id abc-123
  
  # Output as JSON
  python scripts/analyze_timing.py logs/app.log --json
        """
    )
    
    parser.add_argument(
        'log_file',
        type=Path,
        help='Path to log file'
    )
    
    parser.add_argument(
        '--backtest-id',
        type=str,
        help='Filter by specific backtest ID'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"❌ Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)
    
    analysis = analyze_timing_logs(args.log_file, args.backtest_id)
    print_analysis(analysis, args.json)


if __name__ == '__main__':
    main()
