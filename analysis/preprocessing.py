"""Recorded processing profiles; profiles describe distinct historical branches."""
import scanpy as sc
import scanpy.external as sce

PROFILES={
    'early_global':dict(hvg=2000,pca=60,harmony='orig.ident',neighbors=30,pcs=30,resolution=2.0),
    'B':dict(hvg=2000,pca=50,harmony='orig.ident',neighbors=15,pcs=30,resolution=.5),
    'T':dict(hvg=2000,pca=50,harmony='orig.ident',neighbors=25,pcs=30,resolution=2.0),
    'NK':dict(hvg=2000,pca=50,harmony='dataset',neighbors=15,pcs=30,resolution=.2),
    'Myeloid':dict(hvg=2000,pca=50,harmony='orig.ident',neighbors=15,pcs=30,resolution=1.0),
    'july_remerge':dict(hvg=3000,pca=50,harmony='dataset',neighbors=30,pcs=30,resolution=.6),
}
for name,profile in PROFILES.items():
    profile.update(pca_seed=0 if name=='july_remerge' else 97,
                   graph_seed=97 if name in ('early_global','T') else (0 if name=='july_remerge' else None),
                   harmony_seed=97 if name in ('early_global','T','Myeloid') else None,
                   pca_solver='randomized' if name=='early_global' else ('arpack' if name=='july_remerge' else None))


def prepare(x,profile):
    if 'counts' not in x.layers:x.layers['counts']=x.X.copy()
    x.X=x.layers['counts'].copy()
    sc.pp.normalize_total(x,target_sum=10000);sc.pp.log1p(x)
    x.layers['logcounts']=x.X.copy()
    if profile is not PROFILES['july_remerge'] and profile is not PROFILES['early_global']:
        sc.pp.filter_genes(x,min_cells=3)
    try:
        sc.pp.highly_variable_genes(x,flavor='seurat',n_top_genes=profile['hvg'],batch_key='dataset')
    except Exception:
        if profile not in (PROFILES['T'],PROFILES['Myeloid']):raise
        sc.pp.highly_variable_genes(x,flavor='seurat',n_top_genes=profile['hvg'])
    x.raw=x
    work=x[:,x.var.highly_variable].copy()
    scale_kwargs={'max_value':10} if profile is PROFILES['july_remerge'] else {}
    sc.pp.scale(work,zero_center=False,**scale_kwargs)
    kwargs={} if profile['pca_solver'] is None else {'svd_solver':profile['pca_solver']}
    sc.tl.pca(work,n_comps=profile['pca'],random_state=profile['pca_seed'],**kwargs)
    x.obsm['X_pca']=work.obsm['X_pca'].copy()
    return x


def graph(x,profile,basis):
    kwargs={'metric':'cosine'} if profile is PROFILES['july_remerge'] else {}
    seed={} if profile['graph_seed'] is None else {'random_state':profile['graph_seed']}
    sc.pp.neighbors(x,use_rep=basis,n_neighbors=profile['neighbors'],n_pcs=profile['pcs'],**seed,**kwargs)
    sc.tl.leiden(x,resolution=profile['resolution'],**seed)
    kwargs={'min_dist':.35,'spread':1.0} if profile is PROFILES['july_remerge'] else {}
    sc.tl.umap(x,**seed,**kwargs)
    return x


def harmony(x,profile):
    kwargs={} if profile['harmony_seed'] is None else {'random_state':profile['harmony_seed']}
    if profile is PROFILES['july_remerge']:kwargs['max_iter_harmony']=30
    sce.pp.harmony_integrate(x,key=profile['harmony'],basis='X_pca',adjusted_basis='X_pca_harmony',**kwargs)
    return graph(x,profile,'X_pca_harmony')
