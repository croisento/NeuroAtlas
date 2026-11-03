"""Remove reviewer-selected cell IDs; never infer contaminants from HBB alone."""
import pandas as pd
import anndata as ad
from analysis.common import arguments,refuse_overwrite


def main():
    p=arguments(__doc__);p.add_argument('--excluded-cells',required=True,help='CSV with cell_id and reason columns')
    a=p.parse_args();refuse_overwrite(a.output)
    reviewed=pd.read_csv(a.excluded_cells)
    assert {'cell_id','reason'}.issubset(reviewed)
    x=ad.read_h5ad(a.input)
    assert reviewed.cell_id.is_unique and reviewed.reason.notna().all()
    assert reviewed.cell_id.isin(x.obs_names).all(),'Cell list must match this exact input object'
    x=x[~x.obs_names.isin(reviewed.cell_id)].copy()
    x.write_h5ad(a.output,compression='gzip')


if __name__=='__main__':main()
