#!/usr/bin/env python3
"""Audit an MGA route atlas SQLite DB and flag suspicious indexed fields."""
from __future__ import annotations
import argparse, csv, json, sqlite3, math
from pathlib import Path
from typing import Any, Dict, List


def fnum(x, default=None):
    try:
        if x is None or x == "": return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--target')
    ap.add_argument('--min-plausible-tof-days', type=float, default=500.0,
                    help='Flag multi-flyby routes below this TOF as suspicious')
    ap.add_argument('--output-csv')
    ap.add_argument('--output-json')
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    where = []
    params: List[Any] = []
    if args.target:
        where.append('target = ?')
        params.append(args.target)
    sql = 'SELECT * FROM routes' + ((' WHERE ' + ' AND '.join(where)) if where else '')
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    out = []
    counts: Dict[str, int] = {}
    for r in rows:
        seq = r.get('sequence') or ''
        depth = int(r.get('depth') or 0)
        flybys = int(r.get('flyby_count') or 0)
        tof = fnum(r.get('tof_days'))
        flags = []
        if depth >= 3 and tof is not None and tof < args.min_plausible_tof_days:
            flags.append('tof_suspiciously_short_for_multiflyby')
        if tof is None:
            flags.append('missing_tof')
        if r.get('tof_source') and r.get('tof_source') not in ('route_level_days', 'route_level_seconds'):
            flags.append(f"tof_derived:{r.get('tof_source')}")
        if r.get('class') in ('D', None, ''):
            flags.append('low_or_missing_class')
        if fnum(r.get('intermediate_velocity_m_s'), 0) and fnum(r.get('intermediate_velocity_m_s'), 0) > 100:
            flags.append('high_intermediate_velocity')
        for fl in flags:
            counts[fl] = counts.get(fl, 0) + 1
        rr = {k: r.get(k) for k in [
            'route_id','sequence','class','score','tof_days','tof_source','patch_dv_m_s',
            'intermediate_velocity_m_s','final_vinf_m_s','min_rp_margin_km','risk_flags','source_file'
        ] if k in r}
        rr['audit_flags'] = ';'.join(flags)
        out.append(rr)

    out.sort(key=lambda r: (bool(r.get('audit_flags')), r.get('score') if r.get('score') is not None else 1e99))

    print('='*80)
    print('MGA ROUTE ATLAS AUDIT V0.1')
    print('='*80)
    print(f'DB:     {args.db}')
    print(f'Rows:   {len(rows)}')
    print('Flags:')
    if counts:
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f'  {k:<42} {v}')
    else:
        print('  none')
    print('Top rows:')
    for i, r in enumerate(out[:10], 1):
        print(f" {i}. {r.get('sequence')} | class={r.get('class')} | score={r.get('score')} | "
              f"TOF={r.get('tof_days')} ({r.get('tof_source')}) | dv={r.get('patch_dv_m_s')} | "
              f"rpM={r.get('min_rp_margin_km')} | flags={r.get('audit_flags') or '-'}")
    print('='*80)

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            cols = list(out[0].keys()) if out else []
            w = csv.DictWriter(f, fieldnames=cols)
            if cols:
                w.writeheader()
                w.writerows(out)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps({'rows': len(rows), 'flag_counts': counts, 'top': out[:20]}, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
