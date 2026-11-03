"""Figure 3a–f: global and lineage UMAPs and canonical marker dot plot."""
import h5py
import numpy as np
import anndata as ad
import scanpy as sc
from analysis.common import arguments,obs_umap,save_figure,setup_style
from analysis.figures.helpers import discrete_figure,COLORS,ORDER,STYLE
from analysis.expression import selected_logcounts


def main():
    p=arguments(__doc__);p.add_argument('--h5ad-dir',help='Directory containing the four lineage objects')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);setup_style()
    obs,xy=obs_umap(a.input)
    discrete_figure(xy,obs.cell_subtype_short,COLORS,'Cell subtype',a.output/'Figure3a_global_subtypes',True)
    directory=a.input.parent if a.h5ad_dir is None else __import__('pathlib').Path(a.h5ad_dir)
    for letter,lineage,title in [('b','T','T cells'),('c','B','B cells'),('d','ILCS','NK/ILCs'),('e','Myeloid','Myeloid/DCs')]:
        path=directory/f'adata_final_{lineage}_Disease_corrected_min400_except_CART.h5ad'
        with h5py.File(path,'r') as f:
            from anndata.io import read_elem
            local=read_elem(f['obs'])
            basis='X_umap_logcounts_harmony' if lineage!='T' and 'X_umap_logcounts_harmony' in f['obsm'] else 'X_umap'
            coords=f['obsm'][basis][:]
        discrete_figure(coords,local.cell_subtype_short,COLORS,title,a.output/f'Figure3{letter}_{lineage}_subtypes',True)
    markers=STYLE['canonical_markers']
    requested=list(dict.fromkeys(g for group in markers.values() for g in group))
    matrix,obs,var=selected_logcounts(a.input,requested)
    x=ad.AnnData(X=matrix,obs=obs,var=var)
    markers={k:[g for g in v if g in x.var_names] for k,v in markers.items()}
    plot=sc.pl.dotplot(x,markers,groupby='cell_subtype_short',categories_order=[v for v in ORDER if v in obs.cell_subtype_short.unique()],
                       use_raw=False,standard_scale='var',show=False,return_fig=True)
    plot.make_figure();save_figure(plot.fig,a.output/'Figure3f_canonical_marker_dotplot')


if __name__=='__main__':main()
