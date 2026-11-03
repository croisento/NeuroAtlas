"""One-versus-rest Wilcoxon top-50 markers for the released atlas."""
import importlib.util,sys
from analysis.common import arguments,REPOSITORY


def main():
    p=arguments(__doc__);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('atlas_marker_statistics',REPOSITORY/'analysis/source_scripts/67_marker_top50_Wilcoxon_filtered.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.SOURCE=a.input;module.OUT=a.output;module.main()


if __name__=='__main__':main()
