"""Draw the real room boundaries and production camera locations (inspection only)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from data_gen.active_spatial_qa.production_v1 import context, readl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / 'distribution_config.json').read_text())
    rows = readl(root / 'planned_states.jsonl')
    probes = {r['scene_id']: r for r in readl(root / 'failures.jsonl') if r.get('phase') == 'actual_production_gate_probe'}
    index = {}
    for scene in cfg['scene_splits']:
        detector, layout = context(cfg, scene)
        current = [r for r in rows if r['scene_id'] == scene]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.6, 1]})
        for ax in axes:
            for p in layout.room_polys:
                ax.add_patch(Polygon(p, closed=True, edgecolor='#2e5542', facecolor='#e7f2e9', linewidth=1.2))
            for box in detector.object_boxes:
                lo, hi = box.min_point[:2], box.max_point[:2]
                ax.add_patch(Rectangle(lo, *(hi-lo), edgecolor='#b8b8b8', facecolor='none', linewidth=.5))
            for i, row in enumerate(current, 1):
                pose = np.array(row['state_pose_c2w'])
                pos, direction = pose[:2, 3], pose[:2, 2]
                ax.plot(*pos, 'o', color='#2166ac' if row['branch_intent'] == 'Yes' else '#dd7b24', markersize=4)
                ax.arrow(*pos, *(direction*.28), width=.009, head_width=.08, color='#333', alpha=.7)
                if ax is axes[0]:
                    ax.annotate(str(i), pos, xytext=(3, 3), textcoords='offset points', fontsize=7)
            ax.set_aspect('equal', adjustable='datalim')
            ax.autoscale_view()
            ax.set_xlabel('World X (m)')
            ax.set_ylabel('World Y (m)')
            ax.grid(alpha=.15)
        axes[0].set_title(f'{scene}: {len(current)} planned legal cameras\nBlue/orange = intended Yes/No, arrows = view direction')
        probe = probes[scene]
        x, y, _ = probe['position']
        axes[1].plot(x, y, 'rx', markersize=10, mew=2)
        axes[1].annotate('Outside / collision-free\nREJECTED before render\nNo label assigned', (x, y), xytext=(-5, -42), textcoords='offset points', ha='right', fontsize=9)
        axes[1].relim()
        axes[1].autoscale_view()
        axes[1].set_title('Actual production gate probe')
        fig.suptitle('Inspection only: room polygons + expanded collision boxes; not a visibility mask', fontsize=11)
        fig.tight_layout()
        png = output / f'{scene}_positions.png'
        fig.savefig(png, dpi=140)
        plt.close(fig)
        index[scene] = {'plot': str(png.resolve()), 'camera_indices': [{'number': i, 'sample_id': r['sample_id'], 'geometry': r['geometry']} for i, r in enumerate(current, 1)], 'actual_outdoor_probe': probe}
    (output / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({s: v['plot'] for s, v in index.items()}))


if __name__ == '__main__':
    main()
