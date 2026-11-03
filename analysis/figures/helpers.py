"""Plot frozen atlas coordinates without refitting embeddings."""
import json
import numpy as np
import matplotlib.pyplot as plt
from analysis.common import REPOSITORY,save_figure

STYLE=json.loads((REPOSITORY/'analysis/config/figure_style.json').read_text())
ORDER=STYLE['cell_order']
COLORS=STYLE['cell_colors']
DISEASE_COLORS=STYLE['disease_colors']


def umap(ax,xy,values,palette,title,labels=False):
    values=np.asarray(values).astype(str)
    for value in dict.fromkeys(values):
        mask=values==value
        ax.scatter(xy[mask,0],xy[mask,1],s=.5,c=palette.get(value,'#888888'),
                   linewidths=0,rasterized=True,label=value)
        if labels:
            center=np.median(xy[mask],axis=0)
            ax.text(*center,value,fontsize=6,ha='center',va='center')
    ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlabel('UMAP1');ax.set_ylabel('UMAP2')
    ax.spines[['top','right','bottom','left']].set_visible(False)


def discrete_figure(xy,values,palette,title,stem,labels=False):
    fig,ax=plt.subplots(figsize=(8,6))
    umap(ax,xy,values,palette,title,labels)
    if not labels:ax.legend(markerscale=8,frameon=False,fontsize=6,bbox_to_anchor=(1.01,1),loc='upper left')
    save_figure(fig,stem)
