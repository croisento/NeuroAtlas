"""Verify cloud/master alignment and refresh category labels without matrix edits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from anndata.io import read_elem

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'controls without CNS inflammation (unspecific)'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('master', type=Path)
    args = parser.parse_args()
    data = ROOT/'data'
    categories = json.loads((data/'metadata_categories.json').read_text())
    arrays = np.load(data/'atlas_metadata.npz')
    with h5py.File(args.master, 'r') as f:
        obs = read_elem(f['obs'])
        xy = read_elem(f['obsm/X_umap'])
    assert 'disease' not in obs and 'disease' not in categories
    assert len(obs) == len(arrays['X_umap']) == 573569
    assert np.array_equal(xy.astype(np.float32), arrays['X_umap'])
    # Check every categorical annotation, using sample identifiers and GSE to
    # verify cell order, before allowing the two category dictionaries to change.
    changes = []
    for field, levels in categories.items():
        if field == 'PCA_compositional_outlier':
            continue  # Derived from sample-level assessment records, not master obs.
        reference = {'GSM':'orig.ident', 'GSE':'dataset'}.get(field, field)
        expected = obs[reference].astype(str).to_numpy()
        values = np.asarray(levels, dtype=object)[arrays[field+'_codes']]
        updated_levels = [LABEL if v == 'Unknown' and field in ('Disease','Disease1') else v for v in levels]
        updated_values = np.asarray(updated_levels, dtype=object)[arrays[field+'_codes']]
        assert np.array_equal(expected, updated_values), field
        if levels != updated_levels:
            changes.append({'field':field, 'old_label':'Unknown', 'new_label':LABEL,
                            'cells_changed':int(np.count_nonzero(values!=updated_values))})
        categories[field] = updated_levels
    (data/'metadata_categories.json').write_text(json.dumps(categories,ensure_ascii=False,indent=2)+'\n')
    manifest = json.loads((data/'manifest.json').read_text())
    assert 'disease' not in manifest['obs_columns']
    manifest['metadata_source_h5ad'] = args.master.name
    manifest['disease_metadata_updated_utc'] = datetime.now(timezone.utc).isoformat()
    (data/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    report = {'master':args.master.name, 'n_cells':len(obs),
              'categorical_fields_match_master_in_cell_order':True,
              'umap_exactly_matches_master':True, 'changes':changes,
              'expression_and_numeric_qc_arrays_modified':False,
              'atlas_metadata_sha256':hashlib.sha256((data/'atlas_metadata.npz').read_bytes()).hexdigest()}
    (ROOT/'metadata_sync_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
