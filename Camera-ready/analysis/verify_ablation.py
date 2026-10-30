import json,csv,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
old=json.loads((root/'figure_source_logs.json').read_text(encoding='utf-8'))
old=[r for r in old if r['folder'] in ['diag_layernorm_raw_lr_6e-4_long','diag_sphere_bounded_lr_6e-4','diag_sphere_bounded_lr_6e-4_wd0_long']]
logs=old+json.loads((root/'ablation_additional_logs.json').read_text(encoding='utf-8'))
rows=[]; groups={}
for r in logs:
    d=r['data']; s=d['summary']; h=d['history']; e=h.get('epochs',h.get('epoch'))
    assert len(e)==len(h['test_acc'])==len(h['train_acc'])
    assert all(b>a for a,b in zip(e,e[1:]))
    assert all(0<=a<=1 for k in ['test_acc','train_acc'] for a in h[k])
    first=lambda k:next((ep for ep,a in zip(e,h[k]) if a>=.99),None)
    train,test=first('train_acc'),first('test_acc')
    assert train==s['train_epoch_99'] and test==s['test_epoch_99'],r['name']
    assert abs(max(h['test_acc'])-s['peak_test_acc'])<1e-9
    assert s['lr']==.0006
    row=dict(residual=s['model_type'],head='bounded' if s['bounded_head'] else 'raw',wd=s['weight_decay'],seed=s['seed'],train99=train,test99=test,gap99=None if test is None else test-train,peak=max(h['test_acc']),budget=s['max_epochs'],last_epoch=e[-1],intervals=str(sorted(set(b-a for a,b in zip(e,e[1:])))),source=r['url'])
    rows.append(row); groups.setdefault((row['residual'],row['head'],row['wd']),[]).append(row)
assert len(groups)==8 and len(rows)==31
summary=[]
for key,rr in sorted(groups.items(),key=lambda kv:(kv[0][2],kv[0][0],kv[0][1])):
    assert len({r['seed'] for r in rr})==len(rr)
    ok=[r for r in rr if r['test99'] is not None]
    stat=lambda k:(round(statistics.mean([r[k] for r in ok]),3),round(statistics.stdev([r[k] for r in ok]),3)) if len(ok)>1 else (None,None)
    item=dict(residual=key[0],head=key[1],wd=key[2],n=len(rr),success=len(ok),test99=stat('test99'),gap99=stat('gap99'),peak_range=[min(r['peak'] for r in rr),max(r['peak'] for r in rr)],budgets=sorted({r['budget'] for r in rr}),intervals=sorted({r['intervals'] for r in rr}))
    summary.append(item)
with (root/'ablation_verified_per_run.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(root/'ablation_verified_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
