"""Participant-ID registry with original identifiers, internal IDs and evidence."""
from pathlib import Path
import argparse,re,json
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font,Alignment,PatternFill

LABEL_ONLY={'GSE134577','GSE141578','GSE200164','GSE244499'}
REUSED={
 'MS19270':'MS19270','MS49131':'MS49131','MS58637':'MS58637','MS71658':'MS71658',
 'IIH32190':'PTC32190','IIH41540':'PTC41540','IIH45044':'PTC45044','IIH85037':'PTC85037',
}
REUSE_PAPER='https://pmc.ncbi.nlm.nih.gov/articles/PMC7831653/'


def source_id(row):
    title=str(row.GEO_sample_title);gse=row.dataset
    if gse=='GSE133028':return title.split('_')[0],title,'Source participant code in GEO title'
    if gse in ('GSE138266','GSE194078'):return title.rsplit('_',1)[0],title,'Source participant code in GEO title'
    if gse=='GSE163005':return re.search(r'patient#([A-Za-z0-9]+)',title).group(1),title,'Explicit subject ID in GEO title/characteristics'
    if gse=='GSE186802':return str(row.GSM).split('-Day')[0],str(row.GSM),'Patient prefix in source matrix cell labels'
    if gse=='GSE243905':return re.search(r'Control (C\d+)',title).group(1),title,'Source control participant code in GEO title'
    if gse=='GSE247134':return title.removesuffix('_GEX'),title,'Source participant label in GEO title'
    if gse=='GSE200164':return '',title.rsplit('_',1)[1],'Source sample label; separate participant ID not verified'
    if gse=='GSE134577':return '',title.removeprefix('CSF_'),'Source sample label; separate participant ID not verified'
    if gse=='GSE141578':return '',title,'Source sample label; separate participant ID not verified'
    if gse=='GSE244499':return '','5A','Source library/sample label in GEO description; separate participant ID not verified'
    raise ValueError(gse)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final-root',type=Path,required=True)
    a=p.parse_args();root=a.final_root;out=root/'Participant_IDs'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    meta=pd.read_csv(root/'ShinyAppsCloud/data/sample_metadata.csv',keep_default_na=False).rename(columns={'GSE':'dataset'})
    titles=pd.read_csv(root/'Tables/final_sample_audit.csv')[['dataset','GSM','GEO_sample_title','GEO_accession']]
    meta=meta.merge(titles,on=['dataset','GSM'],validate='one_to_one')
    rows=[]
    for row in meta.itertuples():
        original,label,evidence=source_id(row)
        provisional=not bool(original)
        local_key=f'{row.dataset}:{original}' if original else f'{row.dataset}:sample:{row.GSM}'
        canonical=local_key
        reuse_evidence=''
        if row.dataset=='GSE163005' and original in REUSED:
            canonical='GSE138266:'+REUSED[original]
            reuse_evidence='GEO states previously uploaded in GSE138266; Heming et al. report reuse of 4 MS and 4 IIH participants; source codes and demographics agree'
        rows.append(dict(dataset=row.dataset,GSM=row.GSM,GEO_accession=row.GEO_accession,
            GEO_sample_title=row.GEO_sample_title,source_participant_id=original,source_sample_label=label,
            study_participant_key=local_key,canonical_subject_key=canonical,
            ID_status='Atlas-assigned sample-based ID' if provisional else 'Source participant ID available',
            identity_resolution='Provisional sample-based identity' if provisional else 'Source participant identity',
            ID_evidence=evidence,cross_study_evidence=reuse_evidence,
            source_URL=f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={row.GEO_accession}',
            tissue=row.tissue,Disease=row.Disease,Disease1=row.Disease1,age=row.age,gender=row.gender,
            Atlas_cells=row.Atlas_cells,pair_id=row.pair_id,paired_sample=row.paired_sample,
            included_in_paired_analysis=row.included_in_paired_analysis,pairing_status=row.pairing_status))
    samples=pd.DataFrame(rows)
    keys=sorted(samples.canonical_subject_key.unique())
    internal={key:f'NCSF-P{i:04d}' for i,key in enumerate(keys,1)}
    samples.insert(2,'atlas_participant_id',samples.canonical_subject_key.map(internal))
    sample_key=samples.dataset+':'+samples.GSM
    assert sample_key.is_unique and len(samples)==205
    counts=samples.groupby('atlas_participant_id').dataset.nunique()
    samples['cross_study_reuse_flag']=samples.atlas_participant_id.map(counts).gt(1)
    samples.loc[samples.GSM.eq('GSM5264668'),'paired_sample']='GSM5264667'
    records=[]
    for identifier,group in samples.groupby('atlas_participant_id',sort=True):
        assert group.age.astype(str).nunique()==1 and group.gender.nunique()==1,identifier
        assert group.Disease.nunique()==1,identifier
        records.append(dict(atlas_participant_id=identifier,
            source_participant_ids='; '.join(dict.fromkeys(f'{r.dataset}:{r.source_participant_id}' for r in group.itertuples() if r.source_participant_id)),
            source_sample_labels='; '.join(dict.fromkeys(group.source_sample_label)),
            ID_status=group.ID_status.iloc[0],datasets='; '.join(sorted(group.dataset.unique())),
            n_datasets=group.dataset.nunique(),n_sample_records=len(group),
            sample_records='; '.join(f'{r.dataset}:{r.GSM}' for r in group.itertuples()),
            tissues='; '.join(sorted(group.tissue.unique())),Disease=group.Disease.iloc[0],
            age=group.age.iloc[0],gender=group.gender.iloc[0],
            included_paired_sample_records=int(group.included_in_paired_analysis.sum()),
            cross_study_reuse_flag=bool(group.cross_study_reuse_flag.any()),
            interpretation='Internal registry ID; independent participant status unresolved' if group.ID_status.iloc[0].startswith('Atlas-assigned') else 'Source participant identity; shared records linked where documented'))
    participants=pd.DataFrame(records)
    assert participants.atlas_participant_id.is_unique
    assert samples.included_in_paired_analysis.sum()==80
    cross=participants.loc[participants.cross_study_reuse_flag].copy()
    assert len(cross)==8
    summary=samples.groupby('dataset').agg(sample_records=('GSM','size'),
        atlas_IDs_in_study=('atlas_participant_id','nunique'),
        records_with_source_participant_ID=('source_participant_id',lambda x:x.ne('').sum()),
        sample_based_ID_records=('ID_status',lambda x:x.str.startswith('Atlas-assigned').sum()),
        records_in_paired_analysis=('included_in_paired_analysis','sum'),
        records_in_documented_cross_study_groups=('cross_study_reuse_flag','sum')).reset_index()
    definitions=pd.DataFrame([
        ('atlas_participant_id','Atlas-assigned identifier for every record. Source identities share an ID only when supported by source codes or documented reuse.'),
        ('source_participant_id','Original participant code where identified in checked GEO records, source matrix labels or verified pairing records.'),
        ('source_sample_label','Original sample/library label. A sample label is not automatically a separately verified participant ID.'),
        ('Provisional sample-based identity','No distinct participant ID was established from the checked records; one internal ID per sample is used provisionally. This does not prove the source paper contains no additional ID.'),
        ('Study qualification','Numeric or short labels from different studies are not merged without explicit evidence.'),
        ('Cross-study reuse','8 participants shared by GSE138266 and GSE163005 are linked using published reuse, source codes and matching demographics.'),
        ('Paired analysis','The original 40 within-study pairs remain unchanged. Cross-study identity links do not add pairs to this analysis.'),
        ('Patient count','The number of assigned IDs is not a validated count of independent biological participants: unresolved source-sample identities and possible undisclosed cross-study overlap remain.'),
        ('GSE244499','GSM7818261 has sample/library label 5A in GEO; this is retained as a sample label, not promoted to a verified patient ID.'),
    ],columns=['Field','Definition'])
    evidence=pd.DataFrame([
        ('Heming et al. 2021','Control Subjects: 4 IIH and 4 MS records reused from Schafflick et al.',REUSE_PAPER),
        ('GSM4970058','MS49131; previously uploaded in GSE138266','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4970058'),
        ('GSM4970063','IIH45044; previously uploaded in GSE138266','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4970063'),
        ('GSM4970064','IIH85037; previously uploaded in GSE138266','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4970064'),
        ('GSM7818261','Source description lists 5A expression files','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7818261'),
        ('Cantoni et al. 2025','GSM5264668 is a duplicate blood record; exclusion is limited to paired analysis','https://www.jci.org/articles/view/177793'),
    ],columns=['Source','Evidence','URL'])
    samples.to_csv(out/'All_samples_participant_IDs.csv',index=False)
    participants.to_csv(out/'Participant_registry.csv',index=False)
    cross.to_csv(out/'Documented_cross_study_reuse.csv',index=False)
    with pd.ExcelWriter(out/'All_participant_IDs.xlsx',engine='openpyxl') as writer:
        for name,df in [('Sample_IDs',samples),('Participant_registry',participants),('Study_summary',summary),
                        ('Cross_study_reuse',cross),('ID_definitions',definitions),('Evidence_sources',evidence)]:
            df.to_excel(writer,sheet_name=name,index=False)
            sheet=writer.book[name];sheet.freeze_panes='D2';sheet.auto_filter.ref=sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width=min(65,max(16,len(str(column[0].value))+3))
    # Add IDs and pairing information to a new copy of the final counts-based QC workbook.
    source=root/'Table2 Final_all_GSM_QC_audit_FINAL_QC.xlsx'
    target=root/'Table2 Final_all_GSM_QC_audit_FINAL_QC_with_participant_IDs.xlsx'
    if target.exists():raise FileExistsError(target)
    workbook=load_workbook(source);sheet=workbook['Final_sample_QC']
    lookup=samples.set_index(['dataset','GSM'])
    fields=['atlas_participant_id','source_participant_id','source_sample_label','ID_status',
            'study_participant_key','cross_study_reuse_flag','pair_id','paired_sample',
            'included_in_paired_analysis','pairing_status','source_URL']
    initial=sheet.max_column
    for j,field in enumerate(fields,start=initial+1):
        cell=sheet.cell(1,j,field);cell.font=Font(bold=True);cell.fill=PatternFill('solid',fgColor='DDEBF7')
        sheet.column_dimensions[cell.column_letter].width=30
        for i in range(2,sheet.max_row+1):
            key=(sheet.cell(i,1).value,sheet.cell(i,2).value)
            value=lookup.loc[key,field]
            if hasattr(value,'item'):value=value.item()
            sheet.cell(i,j,value if pd.notna(value) else None)
    sheet.auto_filter.ref=sheet.dimensions
    for title,frame in [('Participant_registry',participants),('Participant_ID_notes',definitions),('Cross_study_reuse',cross)]:
        s=workbook.create_sheet(title);s.append(frame.columns.tolist())
        for row in frame.itertuples(index=False,name=None):s.append(row)
    workbook.save(target)
    result={'samples':len(samples),'registry_IDs':len(participants),
        'registry_IDs_with_source_participant_codes':int(participants.ID_status.eq('Source participant ID available').sum()),
        'provisional_sample_based_registry_IDs':int(participants.ID_status.str.startswith('Atlas-assigned').sum()),
        'documented_cross_study_participant_groups':len(cross),
        'sample_records_in_cross_study_groups':int(samples.cross_study_reuse_flag.sum()),
        'paired_analysis_pairs_unchanged':40,'independent_biological_participant_count':'Not established',
        'QC_workbook_with_IDs':str(target)}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print(summary.to_string(index=False))


if __name__=='__main__':main()
