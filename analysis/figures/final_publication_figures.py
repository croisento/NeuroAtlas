"""Recreate the final figure layouts with the later plotting settings.

Final reference PDFs were manually assembled. This module draws data panels
and reconstructs their page layout; it never uses reference PDF artwork as data.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.colors import to_hex, to_rgb
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
from figures.common import metadata_object, qc_object
from figures.annotation_plotting import SOFT_COLOR_MAP, CELL_TYPE_ORDER

UNSPECIFIC = 'controls without CNS inflammation (unspecific)'
DISEASE_ORDER = ['Younger control','Older control','MCI/AD','PD','IIH','MS',
    'Ab-mediated IDD','VE','Neuro-COVID','CAR-T-DIPG','NT1','CIS','RIS','DLB',
    'PHN','Uveitis','NMO','NT2',UNSPECIFIC]
ROOT = Path(__file__).resolve().parents[1]


def setup():
    bold=Path('/home/lishaogang/Arial-Bold.ttf')
    if bold.exists():font_manager.fontManager.addfont(str(bold))
    for candidate in [Path('/home/lishaogang/.fonts/arial.ttf'),Path('/home/lishaogang/Arial-Regular.ttf')]:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            family = font_manager.FontProperties(fname=candidate).get_name()
            break
    else: family='DejaVu Sans'
    plt.rcParams.update({'font.family':family,'font.size':8,'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    return family


def axis(fig, box):
    """Top-left normalised coordinates, mirroring final PDF panel placement."""
    x,y,w,h=box
    return fig.add_axes([x,1-y-h,w,h])


def save(fig, output, name):
    output.mkdir(parents=True,exist_ok=True)
    # Fixed canvas; tight cropping would change the reference page geometry.
    fig.savefig(output/f'{name}.pdf',dpi=400)
    fig.savefig(output/f'{name}.png',dpi=200)
    plt.close(fig)
    if name.startswith('Figure2 '):
        export_figure2_panels(output / (name + '.pdf'), output)


def export_figure2_panels(path, output):
    """Export the panel PDFs and PNGs from the same assembled Figure 2 page."""
    import pymupdf
    panels = {
        'Figure2a_CSF_QC': (0, 0, 720, 232),
        'Figure2b_PBMC_QC': (0, 232, 720, 464),
        'Figure2c_dataset_UMAP': (0, 464, 360, 790),
        'Figure2d_tissue_UMAP': (360, 464, 720, 790),
    }
    with pymupdf.open(path) as source:
        for name, bounds in panels.items():
            clip = pymupdf.Rect(bounds)
            with pymupdf.open() as panel:
                page = panel.new_page(width=clip.width, height=clip.height)
                page.show_pdf_page(page.rect, source, 0, clip=clip)
                panel.save(output / (name + '.pdf'), garbage=4, deflate=True)
                page.get_pixmap(matrix=pymupdf.Matrix(3, 3)).save(output / (name + '.png'))


def umap(ax, xy, labels, order, colors, size=.15, alpha=.72, legend=True):
    labels=np.asarray(labels)
    # Preserve original cell order, as a single Scanpy categorical scatter does.
    # Drawing complete categories successively can obscure earlier tissues.
    ax.scatter(xy[:,0],xy[:,1],s=size,color=[colors[label] for label in labels],
               alpha=alpha,linewidths=0,rasterized=True)
    ax.set(xticks=[],yticks=[],xlabel='UMAP1',ylabel='UMAP2')
    for spine in ax.spines.values():spine.set_linewidth(.5);spine.set_color('#888888')
    if legend:ax.legend(handles=[Line2D([],[],marker='o',ls='',color=colors[d],label=d,ms=3)
        for d in order if np.any(labels==d)],frameon=False,fontsize=10,
        bbox_to_anchor=(1.02,.95),loc='upper left')


def figure2(x, output, input_path):
    cache=output/'Figure2_QC_source.csv.gz'
    identity_file=output/'Figure2_QC_source.input.json'
    stat=input_path.stat()
    identity={'input':str(input_path.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    if cache.exists() and identity_file.exists() and json.loads(identity_file.read_text())==identity:
        frame=pd.read_csv(cache,index_col=0)
        if not frame.index.equals(x.obs_names):raise ValueError('QC cache cell order differs from input')
    else:
        frame=qc_object(input_path).obs
        frame.to_csv(cache)
        identity_file.write_text(json.dumps(identity,indent=2)+'\n')
    style=json.loads((Path(__file__).parent/'figure_style.json').read_text())
    order=list(style['dataset_colors'])
    # Original 33 rounds the desaturated colours to hex; the final PDF stores
    # three-decimal RGB values after Illustrator export.
    palette={d:tuple(round(v,3) for v in to_rgb(to_hex(sns.desaturate(c,.75))))
             for d,c in style['dataset_colors'].items()}
    fig=plt.figure(figsize=(10,790/72))
    keys=['n_genes_by_counts','total_counts','pct_counts_mt','pct_counts_ribo']
    names=['Detected genes per cell','Total counts per cell','Mitochondrial counts (%)','Ribosomal counts (%)']
    for row,tissue in enumerate(['CSF','PBMC']):
        local=frame.loc[x.obs.tissue.astype(str).eq(tissue)]
        present=[d for d in order if local.dataset.eq(d).any()]
        for col,(key,title) in enumerate(zip(keys,names)):
            # Measured from the original final PDF: 132 x 108 pt per QC panel.
            ax=axis(fig,[(34+178*col)/720,(52+220*row)/790,132/720,108/790])
            line_width=[.267,.268][row]
            sns.violinplot(data=local,x='dataset',y=key,hue='dataset',order=present,
                hue_order=present,palette=palette,saturation=1,inner='box',linewidth=line_width,
                inner_kws={'box_width':[1.201,1.205][row],'whis_width':[.400,.402][row]},
                cut=0,legend=False,ax=ax)
            ax.set(xlabel='',ylabel='',title=title)
            ax.set_title(title,fontsize=12,pad=12)
            ax.tick_params(axis='x',length=0,labelbottom=False)
            ax.tick_params(axis='y',labelsize=9,length=0)
            ax.yaxis.set_major_locator(MultipleLocator([1000,10000,2.5,10][col]))
            ax.grid(color='#cccccc',linewidth=line_width);ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(True);spine.set_color('#cccccc');spine.set_linewidth(line_width)
            # Final labels were rotated/repositioned during manual assembly.
            for i,dataset in enumerate(present):
                x_pt=34+178*col+132*(i+.5)/len(present)-4.3066
                y_pt=160+220*row+56.0386
                fig.text(x_pt/720,1-y_pt/790,dataset,rotation=90,
                         rotation_mode='anchor',ha='left',va='baseline',fontsize=9)
        fig.text(12/720,1-(22+220*row)/790,f'{chr(97+row)}  {tissue}',weight='bold',fontsize=13)
    ax=axis(fig,[.048,.65,.32,.307])
    umap(ax,x.obsm['X_umap'],x.obs.dataset.astype(str),order,palette)
    ax.set_title('UMAP by dataset',fontsize=11.5);ax.get_legend().set_title('Dataset',prop={'size':10})
    ax.set_xlabel('UMAP1',fontsize=10);ax.set_ylabel('UMAP2',fontsize=10)
    fig.text(.017,.385,'c',weight='bold',fontsize=13)
    ax=axis(fig,[.548,.65,.32,.307])
    umap(ax,x.obsm['X_umap'],x.obs.tissue.astype(str),['CSF','PBMC'],{'CSF':'#36215f','PBMC':'#8b0902'})
    ax.set_title('UMAP by tissue',fontsize=11.5);ax.get_legend().set_title('Tissue',prop={'size':10})
    ax.set_xlabel('UMAP1',fontsize=10);ax.set_ylabel('UMAP2',fontsize=10)
    fig.text(.517,.385,'d',weight='bold',fontsize=13)
    save(fig,output,'Figure2 Quality control and dataset integration')


def figure4(source, stats, output):
    fig,axes=plt.subplots(1,3,figsize=(15,480/72))
    datasets=['GSE133028','GSE138266','GSE194078','GSE243905']
    colors=dict(zip(datasets,['#377eb8','#4daf4a','#984ea3','#e39d31']))
    for ax,subtype in zip(axes,['NK bright','NK dim','CD4 Memory']):
        data=source.loc[source.cell_subtype_short.eq(subtype)]
        for r in data.itertuples():ax.plot([0,1],[r.CSF_percent,r.PBMC_percent],
            color=colors[r.dataset],alpha=.6,lw=.8,marker='o',ms=3)
        stat=stats.loc[stats.cell_subtype_short.eq(subtype)].iloc[0]
        ax.set(xticks=[0,1],xticklabels=['CSF','PBMC'],xlim=(-.2,1.2),
            ylabel='Within-sample cell proportion (%)',title=f'{subtype}\nBH-adjusted P={stat.p_adjusted_BH:.3g}',
            ylim=(0,max(data.CSF_percent.max(),data.PBMC_percent.max())*1.12+1))
        ax.tick_params(labelsize=12)
        ax.set_ylabel('Within-sample cell proportion (%)',fontsize=12)
        ax.set_title(f'{subtype}\nBH-adjusted P={stat.p_adjusted_BH:.3g}',fontsize=14)
    pairs=source.drop_duplicates('pair_id')
    fig.legend(handles=[Line2D([],[],color=colors[d],marker='o',lw=1,
        label=f'{d} (n={(pairs.dataset==d).sum()} pairs)') for d in datasets],
        loc='lower center',bbox_to_anchor=(.5,.1),ncol=2,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.07,right=.98,top=.86,bottom=.30,wspace=.32)
    save(fig,output,'Figure4 Comparison of immune-cell proportions in donor-matched CSF and PBMC samples')


def figure5(x, output):
    obs=x.obs.copy();obs['Disease']=obs.Disease.astype(str)
    style=json.loads((Path(__file__).parent/'figure_style.json').read_text())
    colors=style['disease_colors'];colors[UNSPECIFIC]=colors.get('Unknown','#ff7f0e')
    order=DISEASE_ORDER.copy()
    # The reference places the former Unknown group third in the disease
    # legend/sample stacks, but last in the pooled-cell bars.
    sample_order=order[:2]+[UNSPECIFIC]+order[2:-1]
    fig=plt.figure(figsize=(1000/72,600/72))
    ax=axis(fig,[.01,.07,.25,.38])
    umap(ax,x.obsm['X_umap'],obs.Disease,sample_order,colors,size=.12)
    ax.set_title('Disease',weight='bold',fontsize=9)
    ax.yaxis.label.set_size(5)
    ax.yaxis.set_label_coords(-.022,.5)
    ax.get_legend().set_visible(False)
    display=lambda d: 'controls without CNS\ninflammation (unspecific)' if d==UNSPECIFIC else d
    handles=[Line2D([],[],ls='',marker='o',color=colors[d],label=display(d),ms=2) for d in sample_order if obs.Disease.eq(d).any()]
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.0,.97),fontsize=5.5,frameon=False)
    samples=obs[['dataset','orig.ident','tissue','Disease']].drop_duplicates()
    for box,tissue,label in [([.355,.08,.395,.23],'CSF','b'),([.81,.08,.16,.23],'PBMC','c')]:
        ax=axis(fig,box);s=samples[samples.tissue.astype(str).eq(tissue)]
        counts=pd.crosstab(s.dataset,s.Disease).reindex(columns=[d for d in sample_order if s.Disease.eq(d).any()],fill_value=0)
        counts=counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
        bottom=np.zeros(len(counts))
        for disease in counts:
            values=counts[disease].to_numpy();ax.bar(np.arange(len(counts)),values,bottom=bottom,color=colors[disease],width=.72)
            for i,(value,b) in enumerate(zip(values,bottom)):
                if value:
                    if value<3:ax.annotate(str(value),(i+.36,b+value/2),xytext=(i+.65,max(b+value/2,1.8)),fontsize=6,va='center',arrowprops={'arrowstyle':'-','lw':.3})
                    else:ax.text(i,b+value/2,str(value),fontsize=6,ha='center',va='center')
            bottom+=values
        for i,n in enumerate(bottom):ax.text(i,n+.8,f'n={int(n)}',ha='center',fontsize=6)
        ax.set_xticks(np.arange(len(counts)),counts.index,rotation=45,ha='right',fontsize=6.5)
        ax.set_ylabel('Number of samples',fontsize=7);ax.tick_params(axis='y',labelsize=6)
        ax.set_ylim(0,55 if tissue=='CSF' else 25);ax.set_title(tissue+' sample composition',fontsize=9,weight='bold')
        counts.to_csv(output/f'Figure5_{tissue}_sample_counts.csv')
    selected=[d for d in sample_order if obs.Disease.eq(d).any()]
    legend_order=[d for col in range(5) for d in selected[col::5]]
    fig.legend(handles=[Patch(color=colors[d],label=display(d)) for d in legend_order],
        loc='upper left',bbox_to_anchor=(.35,.62),ncol=5,fontsize=6,frameon=False,title='Disease / control group',title_fontsize=7)
    panels=[]
    for tissue in ['CSF','PBMC']:
        local=obs[obs.tissue.astype(str).eq(tissue)]
        present=[d for d in order if local.Disease.eq(d).any()]
        counts=pd.crosstab(local.Disease,local.cell_subtype_short).reindex(index=present,columns=CELL_TYPE_ORDER,fill_value=0)
        panels.append(counts)
    total=sum(len(t) for t in panels);left=.05
    for tissue,counts in zip(['CSF','PBMC'],panels):
        width=.76*len(counts)/total
        ax=axis(fig,[left,.545,width,.34]);totals=counts.sum(axis=1);fractions=counts.div(totals,axis=0)
        bottom=np.zeros(len(counts))
        for subtype in CELL_TYPE_ORDER:
            ax.bar(np.arange(len(counts)),fractions[subtype],bottom=bottom,color=SOFT_COLOR_MAP[subtype],width=.78)
            bottom+=fractions[subtype].to_numpy()
        names=[d if d!=UNSPECIFIC else 'controls without CNS\ninflammation (unspecific)' for d in counts.index]
        ax.set_xticks(np.arange(len(counts)),[f'{d}\n(n={totals.iloc[i]:,})' for i,d in enumerate(names)],rotation=55,ha='right',fontsize=6)
        for tick,disease in zip(ax.get_xticklabels(),counts.index):
            if disease==UNSPECIFIC:tick.set_fontsize(4.6)
        ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.tick_params(axis='y',labelsize=5)
        if tissue=='CSF':ax.set_ylabel('Cell-type proportion',fontsize=6)
        else:ax.set_yticks([]);ax.spines['left'].set_visible(False)
        ax.set_title(tissue,fontsize=9,weight='bold');left+=width+.035
    fig.legend(handles=[Patch(color=SOFT_COLOR_MAP[s],label=s) for s in CELL_TYPE_ORDER],
        loc='upper left',bbox_to_anchor=(.875,.48),fontsize=5.5,frameon=False,title='Cell type',title_fontsize=7)
    for letter,pos in [('a',(.01,.965)),('b',(.325,.965)),('c',(.79,.965)),('d',(.017,.48))]:fig.text(*pos,letter,weight='bold',fontsize=8)
    save(fig,output,'Figure5 Sample distribution and immune-cell composition across clinical groups')


def figureS1(x, output):
    fig=plt.figure(figsize=(980/72,825/72));obs=x.obs;xy=x.obsm['X_umap']
    pca_rows=[]
    for tissue,start,width,ncol in [('CSF',.052,.115,4),('PBMC',.708,.115,2)]:
        local=obs.loc[obs.tissue.astype(str).eq(tissue)]
        matrix=pd.crosstab(local['orig.ident'],local.cell_subtype_short)
        matrix=matrix.div(matrix.sum(axis=1),axis=0).fillna(0)
        diseases=(['Younger control','Older control','MCI/AD','PD','IIH','MS','Ab-mediated IDD','VE'] if tissue=='CSF' else ['MS','Ab-mediated IDD','IIH','Younger control'])
        for i,disease in enumerate(diseases):
            ids=sorted(local.loc[local.Disease.astype(str).eq(disease),'orig.ident'].unique())
            if len(ids)<5:continue
            pca=PCA(n_components=2);coords=pca.fit_transform(StandardScaler().fit_transform(matrix.loc[ids]))
            z=zscore(np.sqrt((coords**2).sum(axis=1)),ddof=0);flag=np.nan_to_num(z)>1.5
            for j,sample in enumerate(ids):
                pca_rows.append({'tissue':tissue,'Disease':disease,'sample':sample,
                    'PC1':coords[j,0],'PC2':coords[j,1],'distance_z':z[j],
                    'PCA_compositional_outlier':bool(flag[j])})
            ax=axis(fig,[start+(i%ncol)*.168,.103+(i//ncol)*.25,width,.142])
            ax.scatter(coords[~flag,0],coords[~flag,1],s=3,c='skyblue',alpha=.8,rasterized=True)
            ax.scatter(coords[flag,0],coords[flag,1],s=5,c='red',rasterized=True)
            ax.set_title(disease,fontsize=15.24,pad=12)
            for j in np.flatnonzero(flag):
                neat={'GSM6016390':(.5,.91),'GSM6016354':(.58,.72),'GSM6016361':(.58,.1),
                      'GSM5264639':(.38,.85),'GSM4970074':(.6,.22),
                      'GSM5264662':(.5,.83),'GSM4104135':(.38,.12),'GSM5827396':(.6,.5)}
                if ids[j] in neat:
                    ax.text(*neat[ids[j]],ids[j],transform=ax.transAxes,fontsize=11.43,color='red',ha='center')
                    continue
                xp=np.clip(coords[j,0],np.min(coords[:,0]),np.max(coords[:,0]));yp=coords[j,1]
                ax.annotate(ids[j],(xp,yp),xytext=(-4 if xp>coords[:,0].mean() else 4,4),
                    textcoords='offset points',ha='right' if xp>coords[:,0].mean() else 'left',fontsize=11.43,color='red')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',fontsize=13.61)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',fontsize=13.61)
            ax.tick_params(labelsize=11.43);ax.spines[['top','right']].set_visible(True)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_major_locator(MultipleLocator(1 if np.ptp(coords[:,1])<9 else 2))
            for spine in ax.spines.values():spine.set_linewidth(.5);spine.set_color('#aaaaaa')
    handles=[Line2D([],[],marker='o',ls='',c=c,label=l,ms=3) for c,l in [('skyblue','Normal'),('red','Outlier')]]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.35,.96),ncol=2,frameon=False,fontsize=11.43)
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.82,.96),ncol=2,frameon=False,fontsize=11.43)
    scores=obs.doublet_score.to_numpy(float)
    lower,upper=xy.min(axis=0),xy.max(axis=0)
    padding=(upper-lower)*.04
    for tissue,left in [('CSF',.13),('PBMC',.51)]:
        ax=axis(fig,[left,.64,.31,.285])
        indices=np.flatnonzero(obs.tissue.astype(str).eq(tissue).to_numpy())
        indices=indices[np.argsort(scores[indices],kind='stable')]
        points=ax.scatter(xy[indices,0],xy[indices,1],c=scores[indices],s=.5,
            linewidths=0,cmap='viridis',vmin=0,vmax=1,rasterized=True)
        ax.set(title=f'{tissue} (n={len(indices):,})',xticks=[],yticks=[],
            xlabel='UMAP1',ylabel='UMAP2',
            xlim=(lower[0]-padding[0],upper[0]+padding[0]),
            ylim=(lower[1]-padding[1],upper[1]+padding[1]))
        ax.set_aspect('equal',adjustable='box')
    cax=axis(fig,[.85,.64,.011,.285])
    fig.colorbar(points,cax=cax,label='Scrublet doublet score')
    from figures import figure_S1_doublet_by_tissue as tissue_panel
    tissue_panel.SOURCE=None
    tissue_panel.ATLAS=x
    tissue_panel.OUT=output/'Tissue_panels'
    tissue_panel.TABLES=tissue_panel.OUT
    tissue_panel.main()
    # The standalone entry uses the same metadata and unchanged UMAP coordinates.
    for text,pos in [('a  CSF',(.015,.967)),('b  PBMC',(.677,.967)),('c',(.015,.41))]:fig.text(*pos,text,weight='bold',fontsize=17.42)
    pd.DataFrame(pca_rows).to_csv(output/'FigureS1_PCA_source.csv',index=False)
    save(fig,output,'FigureS1 Compositional-outlier assessment and doublet prediction')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',nargs='+',choices=['2','3','4','5','S1'],default=['2','3','4','5','S1'])
    p.add_argument('--h5ad-dir',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);setup();x=metadata_object(a.input)
    if '2' in a.figures:figure2(x,a.output,a.input)
    if '3' in a.figures:
        from figures.final_figure3 import render
        render(a.input,a.h5ad_dir or a.input.parent,a.output)
    if '4' in a.figures:
        from figures.paired_comparison import calculate_from_atlas
        order=pd.read_csv(ROOT/'data/paired_statistics.csv').cell_subtype_short.tolist()
        source,stats=calculate_from_atlas(a.input,ROOT/'data/sample_metadata.csv',
            ROOT/'data/GEO_sample_titles.csv',a.output/'Figure4_tables',order)
        figure4(source,stats,a.output)
    if '5' in a.figures:figure5(x,a.output)
    if 'S1' in a.figures:figureS1(x,a.output)


if __name__=='__main__':main()
