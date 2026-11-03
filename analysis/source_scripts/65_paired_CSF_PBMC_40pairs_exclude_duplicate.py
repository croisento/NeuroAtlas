#!/usr/bin/env python3
"""Literature-guided exclusion of GSM5264668 from donor-matched analysis only."""
import importlib.util
from pathlib import Path

PROJECT = Path('/home/lishaogang/NEU2026_July')


def main():
    spec = importlib.util.spec_from_file_location('paired_analysis', PROJECT / 'Code/61_paired_CSF_PBMC_composition_analysis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TABLES = PROJECT / 'final/Tables/Paired_CSF_PBMC_40pairs_deduplicated'
    module.OUT = PROJECT / 'final/Figures/Paired_CSF_PBMC_40pairs_deduplicated'
    module.main(excluded_samples=('GSM5264668',), expected_pairs=40)


if __name__ == '__main__':
    main()
