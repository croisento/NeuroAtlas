"""Export the current manuscript's data panels using existing atlas embeddings."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import shutil


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--h5ad-dir', type=Path)
    parser.add_argument('--output', type=Path, default=root/'Outputs')
    parser.add_argument('--figures', nargs='+', choices=['2','3','4','5','S1'],
                        default=['2','3','4','5','S1'])
    parser.add_argument('--schematic', type=Path,
                        help='Supplied final Figure1 PDF, copied unchanged')
    args = parser.parse_args()
    output = args.output.resolve()
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    for number in args.figures:
        destination = output / ('Figure' + number)
        command = [sys.executable, '-m', 'figures.final_publication_figures',
                   '--input', str(args.input.resolve()), '--output', str(destination),
                   '--figures', number]
        if args.h5ad_dir:
            command.extend(['--h5ad-dir', str(args.h5ad_dir.resolve())])
        subprocess.run(command, cwd=root, env=env, check=True)
    if args.schematic:
        destination = output / 'Figure1'
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.schematic,
                     destination/'Figure1 Schematic overview of atlas construction.pdf')



if __name__ == '__main__':
    main()
