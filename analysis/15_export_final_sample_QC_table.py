"""Recalculate sample QC from final retained counts and preserve GSM-linked notes."""
from pathlib import Path
import argparse,json,copy
import h5py,numpy as np,pandas as pd
from scipy import sparse
from anndata.io import read_elem
from openpyxl import load_workbook,Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Font,PatternFill,Alignment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--table',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    workbook=load_workbook(a.table)
    original=workbook.worksheets[0]
    headers=[c.value for c in original[1]]
    old=pd.DataFrame(list(original.values)[1:],columns=headers)
    assert not old.duplicated(['dataset','GSM']).any()
    yellow=['Disease Correction','Mechanisms','Disease Classification','Clinical Manifestation']
    clinical=old.set_index(['dataset','GSM'])[yellow]
    with h5py.File(a.input,'r') as f:
        obs=read_elem(f['obs']);var=read_elem(f['var'])
        keys=pd.MultiIndex.from_arrays([obs.dataset.astype(str),obs['orig.ident'].astype(str)],names=['dataset','GSM'])
        assert keys.isin(clinical.index).all() and clinical.index.isin(keys).all()
        node=f['layers/counts'];n,g=map(int,node.attrs['shape'])
        assert n==len(obs)==573569 and g==12890
        mito=var.index.str.upper().str.startswith('MT-')
        ribo=var.index.str.upper().str.startswith(('RPL','RPS'))
        counts=np.zeros(n);detected=np.zeros(n,dtype=np.int32);mt=np.zeros(n);rb=np.zeros(n)
        for first in range(0,n,5000):
            last=min(first+5000,n);ptr=node['indptr'][first:last+1]
            start,end=int(ptr[0]),int(ptr[-1]);values=node['data'][start:end]
            assert np.isfinite(values).all() and (values>=0).all()
            assert np.allclose(values,np.rint(values)), 'counts must be unnormalized integer-valued counts'
            block=sparse.csr_matrix((values,node['indices'][start:end],ptr-start),shape=(last-first,g))
            block.sum_duplicates();block.eliminate_zeros()
            counts[first:last]=np.asarray(block.sum(axis=1)).ravel()
            detected[first:last]=np.diff(block.indptr)
            mt[first:last]=np.asarray(block[:,mito].sum(axis=1)).ravel()
            rb[first:last]=np.asarray(block[:,ribo].sum(axis=1)).ravel()
            if first==0 or last%50000==0 or last==n:print(f'Calculated {last:,}/{n:,} cells',flush=True)
    assert (counts>0).all()
    mt=mt/counts*100;rb=rb/counts*100
    frame=pd.DataFrame({'dataset':obs.dataset.astype(str).to_numpy(),'GSM':obs['orig.ident'].astype(str).to_numpy(),
        'n_genes_by_counts':detected,'total_counts':counts,'pct_counts_mt':mt,'pct_counts_ribo':rb})
    group=frame.groupby(['dataset','GSM'],sort=False)
    summary=group.agg(cells_final=('GSM','size'),median_n_genes_by_counts=('n_genes_by_counts','median'),
        median_total_counts=('total_counts','median'),median_pct_counts_mt=('pct_counts_mt','median'),
        median_pct_counts_ribo=('pct_counts_ribo','median'))
    # Do not silently take the first conflicting sample-level annotation.
    annotated=obs[['dataset','orig.ident','Disease','Disease1','age','gender','tissue']].copy()
    annotated=annotated.rename(columns={'orig.ident':'GSM'})
    for col in annotated:annotated[col]=annotated[col].astype(str)
    for col in ['Disease','Disease1','age','gender','tissue']:
        assert annotated.groupby(['dataset','GSM'])[col].nunique().eq(1).all(),col
    metadata=annotated.drop_duplicates(['dataset','GSM']).set_index(['dataset','GSM'])
    index=pd.MultiIndex.from_frame(old[['dataset','GSM']])
    result=old.iloc[:,:12].copy().set_index(['dataset','GSM'])
    for col in ['Disease','Disease1','age','gender','tissue']:
        result[col]=metadata.loc[result.index,col]
    result=result.join(summary,validate='one_to_one').reset_index()
    assert len(result)==205 and result.cells_final.sum()==573569
    # Highlighted clinical columns remain attached to exactly the original keys.
    pd.testing.assert_frame_equal(result.set_index(['dataset','GSM'])[yellow],clinical)
    # Retain the original workbook sheet intact as the historical QC record.
    original.title='Historical_uniform_QC'
    sheet=workbook.create_sheet('Final_sample_QC',0)
    sheet.append(result.columns.tolist())
    source_column={name:i+1 for i,name in enumerate(headers)}
    original_rows={(original.cell(i,1).value,original.cell(i,2).value):i for i in range(2,original.max_row+1)}
    for row in result.itertuples(index=False,name=None):
        sheet.append([None if pd.isna(v) else v for v in row])
    for j,name in enumerate(result,start=1):
        source_j=source_column.get(name)
        target=sheet.cell(1,j)
        if source_j:
            source=original.cell(1,source_j)
            target._style=copy.copy(source._style)
        target.font=Font(bold=True)
        target.alignment=Alignment(wrap_text=True,vertical='top')
        sheet.column_dimensions[target.column_letter].width=55 if name in yellow else (30 if name=='Disease1' else 23)
        for i,row in enumerate(result.itertuples(index=False),start=2):
            cell=sheet.cell(i,j)
            if source_j:
                source=original.cell(original_rows[(row.dataset,row.GSM)],source_j)
                cell._style=copy.copy(source._style)
            cell.alignment=Alignment(wrap_text=True,vertical='top')
            if name.startswith('median_'):cell.number_format='0.00' if 'pct_' in name else '0.0#'
    sheet.freeze_panes='D2';sheet.auto_filter.ref=sheet.dimensions
    for name in ['median_n_genes_by_counts','median_total_counts','median_pct_counts_mt','median_pct_counts_ribo']:
        j=list(result.columns).index(name)+1
        sheet.cell(1,j).comment=Comment('Recalculated from the final retained cells and layers[counts], over the 12,890 retained genes. Predicted doublets are included. See QC_definitions.','QC source')
    comparison=old[['dataset','GSM','tissue']].copy()
    for key in ['median_n_genes_by_counts','median_total_counts','median_pct_counts_mt','median_pct_counts_ribo']:
        comparison[key+'_uniform_QC']=old[key]
        comparison[key+'_final_counts']=summary.loc[index,key].to_numpy()
    compare=workbook.create_sheet('QC_stage_comparison');compare.append(comparison.columns.tolist())
    for row in comparison.itertuples(index=False,name=None):compare.append(row)
    ranges=[]
    for tissue,df in result.groupby('tissue'):
        for metric in ['median_n_genes_by_counts','median_total_counts','median_pct_counts_mt','median_pct_counts_ribo']:
            lo=df[metric].idxmin();hi=df[metric].idxmax()
            ranges.append(dict(tissue=tissue,n_samples=len(df),metric=metric,minimum=float(df.loc[lo,metric]),
                               minimum_GSM=df.loc[lo,'GSM'],maximum=float(df.loc[hi,metric]),maximum_GSM=df.loc[hi,'GSM']))
    bounds=pd.DataFrame(ranges);s=workbook.create_sheet('QC_ranges');s.append(bounds.columns.tolist())
    for row in bounds.itertuples(index=False,name=None):s.append(row)
    definitions=[
        ('Input object',a.input.name),('Retained cells',n),('Retained genes',g),
        ('Doublets','Predicted doublets retained; no additional cell or gene filtering applied'),
        ('Join key','dataset + GSM; original row order retained; highlighted columns preserved verbatim'),
        ('cells_final','Number of cells currently present per sample'),
        ('n_genes_by_counts','Number of strictly positive raw counts per cell across 12,890 retained genes'),
        ('total_counts','Sum of layers[counts] across 12,890 retained genes per cell'),
        ('pct_counts_mt',f'100 * counts in {int(mito.sum())} MT- genes / total_counts'),
        ('pct_counts_ribo',f'100 * counts in {int(ribo.sum())} RPL/RPS genes / total_counts'),
        ('Sample metrics','Median of the corresponding per-cell values in each sample'),
        ('Historical_uniform_QC','Original input sheet; earlier uniform-QC branch, not final-cell QC'),
        ('Historical removed-cell counts','Available only in Historical_uniform_QC; cannot be reconstructed from retained cells alone'),
        ('Gene universe','These recomputed metrics use the retained 12,890-gene counts matrix; original full-gene metrics in obs may differ'),
    ]
    defs=workbook.create_sheet('QC_definitions');defs.append(['Field','Definition'])
    for row in definitions:defs.append(row)
    defs.column_dimensions['A'].width=32;defs.column_dimensions['B'].width=110
    diagnostics=frame.assign(low_genes=detected<400,high_mt=mt>=20).groupby(['dataset','GSM']).agg(
        final_cells_with_fewer_than_400_shared_genes=('low_genes','sum'),
        final_cells_with_mt_at_least_20_percent=('high_mt','sum')).reset_index()
    diagnostics_sheet=workbook.create_sheet('Final_QC_diagnostics');diagnostics_sheet.append(diagnostics.columns.tolist())
    for row in diagnostics.itertuples(index=False,name=None):diagnostics_sheet.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True);workbook.save(a.output)
    # Round-trip check keys and values after Excel serialization.
    read=pd.read_excel(a.output,sheet_name='Final_sample_QC')
    pd.testing.assert_frame_equal(read.set_index(['dataset','GSM'])[yellow].fillna('<MISSING>'),
                                  clinical.fillna('<MISSING>'),check_dtype=False)
    assert read.cells_final.sum()==573569
    result.to_csv(a.output.with_suffix('.csv'),index=False)
    bounds.to_csv(a.output.with_name(a.output.stem+'_ranges.csv'),index=False)
    validation={'input':str(a.input),'original_table':str(a.table),'output':str(a.output),
        'cells':n,'genes':g,'samples':len(result),'highlighted_columns':yellow,
        'highlighted_columns_GSM_alignment_verified':True,'expression_source':'layers/counts',
        'additional_filtering':False,'ranges':ranges,
        'retained_cells_with_fewer_than_400_shared_genes':int((detected<400).sum()),
        'retained_cells_with_mt_at_least_20_percent':int((mt>=20).sum())}
    a.output.with_name(a.output.stem+'_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(bounds.to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in validation.items() if k!='ranges'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
