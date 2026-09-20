"""Read-only inference diagnostics; never change the production scoring path."""
import numpy as np
import torch
from .streaming_graph_logica import encode_packed, ligand_condition
from .model import _mean_log_probability


def spearman(x,y):
    """Average ranks with ties; no SciPy dependency on the cluster."""
    def ranks(values):
        values=np.asarray(values); order=np.argsort(values,kind="stable")
        out=np.empty(len(values),dtype=float); sorted_values=values[order]
        starts=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1]
        ends=np.r_[starts[1:],len(values)]
        for start,end in zip(starts,ends): out[order[start:end]]=(start+end-1)/2
        return out
    a,b=ranks(x),ranks(y)
    if len(a)<2 or np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])


def score_components(model,ph,dh,da,di,ds,pi,pockets,batch,*,drop_contacts=False,drop_cross=False,use_graph=True):
    """Exact normal-path decomposition plus explicitly named edge interventions.

    Removing contacts still leaves base cross-attention. Removing cross-attention
    still leaves graph contacts. Neither intervention alone is target-free.
    """
    c=model.conditioner; ligand_base,owner=encode_packed(model,batch)
    protein_scores=[]; drug_scores=[]
    for pocket in pockets:
        tokens=pocket.residue_indices+1; direct=pocket.direct_mask.nonzero(as_tuple=True)[0]
        p=ph[:,tokens].expand(batch.batch_size,-1,-1)
        if use_graph:
            pocket_base=c.pocket_encoder(pocket.node_features,pocket.edge_index,pocket.edge_features)
            pocket_nodes=c.protein_context_norm(pocket_base[None]+c.protein_context_in(p))
            flat=pocket_nodes.reshape(-1,pocket_nodes.shape[-1])
            if drop_contacts:
                p_aggregate=torch.zeros_like(flat); l_aggregate=torch.zeros_like(ligand_base)
            else:
                ai=torch.arange(len(ligand_base),device=owner.device).repeat_interleave(len(direct))
                local=direct.repeat(len(ligand_base)); pii=owner[ai]*len(tokens)+local
                features=ligand_base.new_zeros((len(ai),5)); features[:,-1]=1
                to_p=c.ligand_to_pocket(torch.cat([ligand_base[ai],flat[pii],features],-1))
                to_l=c.pocket_to_ligand(torch.cat([flat[pii],ligand_base[ai],features],-1))
                p_aggregate=torch.zeros_like(flat).index_add(0,pii,to_p)
                l_aggregate=torch.zeros_like(ligand_base).index_add(0,ai,to_l)
            pocket_nodes=c.pocket_norm(flat+p_aggregate).reshape_as(pocket_nodes)
            ligand_nodes=c.ligand_norm(ligand_base+l_aggregate)
            p=p+torch.sigmoid(c.protein_graph_gate)*c.protein_out(pocket_nodes)
            d=ligand_condition(model,dh,ligand_nodes,batch.node_offsets)
        else: d=dh
        p=p[:,direct]
        if drop_cross:
            plog=model.base.esm.lm_head(p); dlog=model.base.drug_encoder.lm_head(d)
        else:
            plog,dlog,_,_=model.base.forward_from_hidden(p,torch.ones(p.shape[:2],device=p.device),d,da)
        pids=pi[:,tokens[direct]].expand(batch.batch_size,-1)
        protein_scores.append(_mean_log_probability(plog.float(),pids,torch.ones_like(pids,dtype=torch.bool)))
        drug_scores.append(_mean_log_probability(dlog.float(),di,ds))
    ps=torch.stack(protein_scores).mean(0); dscores=torch.stack(drug_scores).mean(0)
    alpha=model.base.pair_alpha()
    return {"full":alpha*ps+(1-alpha)*dscores,"protein_component":ps,"ligand_component":dscores}


def ligand_only_scores(model,dh,di,selected,batch):
    nodes,_=encode_packed(model,batch)
    graph_hidden=ligand_condition(model,dh,nodes,batch.node_offsets)
    return {
        "ligand_lm_only":_mean_log_probability(model.base.drug_encoder.lm_head(dh).float(),di,selected),
        "ligand_graph_only":_mean_log_probability(model.base.drug_encoder.lm_head(graph_hidden).float(),di,selected)}


def pair_diagnostics(rows, scores):
    """Competition contrasts on a single original development batch."""
    y=rows["proxy_0"].to_numpy(); w=rows["confidence_0"].to_numpy()
    t=rows["count_PGK2"].to_numpy()/np.maximum(rows["source_rows"].to_numpy(),1)
    i=rows["count_PGK2_with_inhibitor"].to_numpy()/np.maximum(rows["source_rows"].to_numpy(),1)
    a,b=np.triu_indices(len(rows),1)
    keep=(np.abs(y[a]-y[b])>=np.log(2)) & (w[a]>0) & (w[b]>0)
    a,b=a[keep],b[keep]; weight=np.minimum(w[a],w[b]); direction=np.sign(y[a]-y[b])
    winner=np.where(direction>0,a,b)
    masks={"all":np.ones(len(a),dtype=bool),"equal_target_counts":t[a]==t[b],
        "both_proxy_nonpositive":(y[a]<=0)&(y[b]<=0),"preferred_target_at_most_one":t[winner]<=1,
        "either_target_at_least_five":(t[a]>=5)|(t[b]>=5),
        "both_target_at_least_five":(t[a]>=5)&(t[b]>=5)}
    result={}
    predictors={**scores,"target_count_only":np.log1p(t),"inverse_inhibitor_count_only":-np.log1p(i)}
    for label,mask in masks.items():
        row={"pairs":int(mask.sum()),"weight_sum":float(weight[mask].sum()),"weighted_wins":{}}
        for name,pred in predictors.items():
            delta=(np.asarray(pred)[a]-np.asarray(pred)[b])*direction
            wins=(delta>0)+.5*(delta==0)
            row["weighted_wins"][name]=float((weight[mask]*wins[mask]).sum())
        result[label]=row
    return result
