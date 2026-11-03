"""Export the dataset, sample and marker tables used by the resource."""
import argparse,shutil
from pathlib import Path
from analysis.common import REPOSITORY


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sources={
        'dataset_sources.csv':REPOSITORY/'analysis/data/dataset_sources.csv',
        'sample_metadata.csv':REPOSITORY/'data/sample_metadata.csv',
        'cell_type_markers.csv':REPOSITORY/'data/cell_type_top50_marker_catalog.csv',
        'cell_subtype_markers.csv':REPOSITORY/'data/cell_subtype_short_top50_marker_catalog.csv',
    }
    for name,path in sources.items():shutil.copy2(path,a.output/name)


if __name__=='__main__':main()
