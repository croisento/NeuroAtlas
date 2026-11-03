"""Run sample-wise Scrublet with the recorded parameters; annotate, retain cells."""
import importlib.util,sys
from analysis.common import arguments,REPOSITORY


def main():
    p=arguments(__doc__);a=p.parse_args()
    source=REPOSITORY/'analysis/source_scripts/49_identify_doublets_final_atlas_no_removal.py'
    spec=importlib.util.spec_from_file_location('atlas_sample_scrublet',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.SOURCE=a.input;module.OUT=a.output
    module.TARGET=a.output/'atlas_scrublet_identified.h5ad'
    module.main()


if __name__=='__main__':main()
