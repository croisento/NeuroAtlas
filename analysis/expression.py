"""Read selected gene columns from CSR logcounts without loading full matrices."""
import h5py,numpy as np
from scipy import sparse
from anndata.io import read_elem


def selected_logcounts(path,requested):
    with h5py.File(path,'r') as f:
        obs=read_elem(f['obs']);var=read_elem(f['var'])
        genes=[gene for gene in requested if gene in var.index]
        columns=var.index.get_indexer(genes)
        node=f['layers/logcounts'];n,p=map(int,node.attrs['shape'])
        mapping=np.full(p,-1,dtype=np.int32);mapping[columns]=np.arange(len(columns))
        blocks=[]
        for first in range(0,n,5000):
            last=min(n,first+5000);ptr=node['indptr'][first:last+1]
            start,end=int(ptr[0]),int(ptr[-1]);ix=node['indices'][start:end]
            keep=mapping[ix]>=0;prefix=np.r_[0,np.cumsum(keep)]
            blocks.append(sparse.csr_matrix((node['data'][start:end][keep],mapping[ix[keep]],prefix[ptr-start]),shape=(last-first,len(genes))))
    return sparse.vstack(blocks,format='csr'),obs,var.loc[genes]
