"""Final Figure 3 page assembly: global map, later32 lineage maps, marker dotplot."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from figures.common import metadata_object
from figures import annotation_plotting as old
from figures import later_lineage_plotting as late
from analysis.expression import selected_logcounts


def render(source, directory, output):
    parts=output/'Figure3_parts';parts.mkdir(parents=True,exist_ok=True)
    font=Path('/home/lishaogang/.fonts/arial.ttf')
    if not font.exists():font=Path('/home/lishaogang/Arial-Regular.ttf')
    late.ARIAL_PATH=font;late.OUTPUT_DIR=parts
    family=late.setup_style();old.ARIAL_FAMILY=family
    x=metadata_object(source)
    order=old.make_ordered_categories(x,'cell_subtype_short',old.CELL_TYPE_ORDER,old.SOFT_COLOR_MAP)
    positions=old.get_group_label_positions(x,'cell_subtype_short')
    fig,ax=plt.subplots(figsize=(8,8))
    sc.pl.umap(x,color='cell_subtype_short',size=3,alpha=.75,ax=ax,show=False,
               frameon=False,legend_loc=None,title=None)
    ax.set_title('')
    old.rasterize_scatter_points(ax)
    np.random.seed(0)
    old.add_celltype_labels(ax,positions,order,label_nudge=old.LABEL_NUDGE,
        fontsize=10,fontfamily=family,stroke_width=3.2,adjust_expand=(1.12,1.22),
        adjust_force=(.25,.45),arrow_lw=.75,arrow_alpha=.45)
    old.add_umap_axis_arrows(ax,arrow_x=.06,arrow_y=.06,arrow_len=.15,lw=2,
                            label_fontsize=10,fontfamily=family)
    old.remove_axis_traces(ax)
    global_pdf=parts/'global_cell_subtypes.pdf'
    fig.savefig(global_pdf,bbox_inches='tight',transparent=True,dpi=400);plt.close(fig)
    maps={}
    for name,tag,basis in [('T','T','X_umap'),('B','B','X_umap_logcounts_harmony'),
                          ('ILCs','ILCS','X_umap_logcounts_harmony'),('Myeloid','Myeloid','X_umap_logcounts_harmony')]:
        path=directory/f'adata_final_{tag}_Disease_corrected_min400_except_CART.h5ad'
        import h5py
        with h5py.File(path,'r') as handle:
            from anndata.io import read_elem
            labels=read_elem(handle['obs'])['cell_subtype_short'].astype(str)
        config={'input':path,'preferred_basis':basis,'types':set(labels),'output':name+'_subtypes'}
        np.random.seed(0)
        late.plot_lineage(name,config,family,400)
        maps[name]=parts/(name+'_subtypes.pdf')
    dot=parts/'canonical_marker_dotplot.pdf'
    genes=list(dict.fromkeys(g for group in old.MARKER_DICT.values() for g in group))
    matrix,obs,var=selected_logcounts(source,genes)
    obj=ad.AnnData(matrix,obs=obs,var=var);obj.layers['logcounts']=obj.X.copy()
    old.plot_marker_gene_dotplot(obj,old.MARKER_DICT,old.DOTPLOT_CELL_TYPE_ORDER,
                                old.DOTPLOT_EXCLUDE_CELLTYPES,output_prefix=str(dot.with_suffix('')))
    # PDF assembly preserves vector text and rasterized cell clouds.
    import pymupdf as pdf
    doc=pdf.open();page=doc.new_page(width=720,height=752)
    panels=[(global_pdf,(10,38,342,365)),(maps['T'],(364,39,524,190)),
            (maps['B'],(545,39,707,190)),(maps['ILCs'],(364,222,524,372)),
            (maps['Myeloid'],(545,222,707,372)),(dot,(24,421,707,725))]
    for path,rect in panels:
        with pdf.open(path) as source_doc:page.show_pdf_page(pdf.Rect(rect),source_doc,0)
    if font.exists():page.insert_font(fontname='AtlasArial',fontfile=str(font));fontname='AtlasArial'
    else:fontname='helv'
    bold=Path('/home/lishaogang/Arial-Bold.ttf')
    if bold.exists():page.insert_font(fontname='AtlasBold',fontfile=str(bold));titlefont='AtlasBold'
    else:titlefont=fontname
    for text,point in [('a',(8,21)),('b',(363,21)),('c',(544,21)),('d',(363,206)),('e',(544,206)),('f',(8,403))]:
        page.insert_text(point,text,fontname=fontname,fontsize=8)
    for text,rect in [('Cell type',(10,10,342,30)),('T cells',(364,10,524,30)),('B cells',(545,10,707,30)),
        ('NK / ILCs',(364,194,524,214)),('Myeloid cells',(545,194,707,214)),('Marker gene expression',(24,390,707,410))]:
        page.insert_textbox(pdf.Rect(rect),text,fontname=titlefont,fontsize=8,align=1)
    path=output/'Figure3 Cell-type annotation of the integrated atlas.pdf'
    doc.save(path,garbage=4,deflate=True)
    page.get_pixmap(matrix=pdf.Matrix(2,2)).save(path.with_suffix('.png'))
    doc.close()
