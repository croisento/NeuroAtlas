"""Historical cell QC: >=400 genes, mitochondrial fraction <20%, genes >=3 cells.

This is the early annotation branch, not a claim that the later shared-count
reconstruction applied min_cells=3 to every input dataset.
"""
import anndata as ad
import scanpy as sc
from analysis.common import arguments,refuse_overwrite


def main():
    p=arguments(__doc__);a=p.parse_args();refuse_overwrite(a.output)
    x=ad.read_h5ad(a.input)
    if 'counts' in x.layers:x.X=x.layers['counts'].copy()
    x.var['mt']=x.var_names.str.upper().str.startswith('MT-')
    sc.pp.calculate_qc_metrics(x,qc_vars=['mt'],percent_top=None,inplace=True)
    x=x[(x.obs.n_genes_by_counts>=400)&(x.obs.pct_counts_mt<20)].copy()
    sc.pp.filter_genes(x,min_cells=3)
    x.write_h5ad(a.output,compression='gzip');print(x)


if __name__=='__main__':main()
