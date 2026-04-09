from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from core.cartridge import CartridgeSpec
from core.token import Token, TokenClass, TokenStream, TokenAgency
from generator.connection_sampler import ConnectionSampler, ConnectionExample
from trainer.connection_model import MysteryConnectionModel
from trainer.loss import EdgeCoherenceLoss


# ---------------------------------------------------------------------------
# Mapping helpers (mirrors train_mystery.py)
# ---------------------------------------------------------------------------

def _build_mappings(tokens: List[Token]):
    id_to_idx = {t.id: i for i, t in enumerate(tokens)}
    class_to_idx = {c.value: i for i, c in enumerate(TokenClass)}
    stream_to_idx = {s.value: i for i, s in enumerate(TokenStream)}
    agency_to_idx = {a.value: i for i, a in enumerate(TokenAgency)}
    return id_to_idx, class_to_idx, stream_to_idx, agency_to_idx


def _pad_sequences(
    seqs: List[List[int]], pad: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(s) for s in seqs), default=1)
    out = np.full((len(seqs), max_len), pad, dtype=np.int64)
    mask = np.ones((len(seqs), max_len), dtype=bool)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        mask[i, : len(s)] = False
    return torch.tensor(out), torch.tensor(mask)


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def _batchify_examples(
    examples: List[ConnectionExample],
    id_to_idx,
    class_to_idx,
    stream_to_idx,
    agency_to_idx,
    device: str,
) -> Dict[str, torch.Tensor]:
    # Per-example accumulators
    ea_ids, ea_cls, ea_str, ea_agt = [], [], [], []
    er_ids_list = []
    eb_ids, eb_cls, eb_str, eb_agt = [], [], [], []
    atm_ids_list, atm_cls_list, atm_str_list = [], [], []
    atm_pos_list: List[List[List[float]]] = []
    ca_ids, ca_cls, ca_str, ca_agt = [], [], [], []
    cr_ids_list = []
    cb_ids, cb_cls, cb_str, cb_agt = [], [], [], []
    targets: List[float] = []

    for ex in examples:
        # ---- Theory edges ------------------------------------------------
        ta, tc, ts, tg, tr, tb, tbc, tbs, tbg = ([] for _ in range(9))
        for conn in ex.theory_edges:
            ta.append(id_to_idx[conn.token_a.id])
            tc.append(class_to_idx[conn.token_a.token_class.value])
            ts.append(stream_to_idx[conn.token_a.stream.value])
            tg.append(agency_to_idx.get(conn.token_a.agency.value, 0))
            tr.append(int(conn.relation))
            tb.append(id_to_idx[conn.token_b.id])
            tbc.append(class_to_idx[conn.token_b.token_class.value])
            tbs.append(stream_to_idx[conn.token_b.stream.value])
            tbg.append(agency_to_idx.get(conn.token_b.agency.value, 0))
        ea_ids.append(ta)
        ea_cls.append(tc)
        ea_str.append(ts)
        ea_agt.append(tg)
        er_ids_list.append(tr)
        eb_ids.append(tb)
        eb_cls.append(tbc)
        eb_str.append(tbs)
        eb_agt.append(tbg)

        # ---- Atmosphere --------------------------------------------------
        ai, ac, as_, ap = [], [], [], []
        for t in ex.atmosphere_tokens:
            ai.append(id_to_idx[t.id])
            ac.append(class_to_idx[t.token_class.value])
            as_.append(stream_to_idx[t.stream.value])
            ap.append([0.0, 0.0])
        atm_ids_list.append(ai)
        atm_cls_list.append(ac)
        atm_str_list.append(as_)
        atm_pos_list.append(ap)

        # ---- Candidate ---------------------------------------------------
        cand = ex.candidate_edge
        ca_ids.append(id_to_idx[cand.token_a.id])
        ca_cls.append(class_to_idx[cand.token_a.token_class.value])
        ca_str.append(stream_to_idx[cand.token_a.stream.value])
        ca_agt.append(agency_to_idx.get(cand.token_a.agency.value, 0))
        cr_ids_list.append(int(cand.relation))
        cb_ids.append(id_to_idx[cand.token_b.id])
        cb_cls.append(class_to_idx[cand.token_b.token_class.value])
        cb_str.append(stream_to_idx[cand.token_b.stream.value])
        cb_agt.append(agency_to_idx.get(cand.token_b.agency.value, 0))
        targets.append(ex.target_coherence)

    # ---- Pad theory edges ------------------------------------------------
    ea_t, edge_mask = _pad_sequences(ea_ids)
    ea_cls_t, _ = _pad_sequences(ea_cls)
    ea_str_t, _ = _pad_sequences(ea_str)
    ea_agt_t, _ = _pad_sequences(ea_agt)
    er_t, _ = _pad_sequences(er_ids_list)
    eb_t, _ = _pad_sequences(eb_ids)
    eb_cls_t, _ = _pad_sequences(eb_cls)
    eb_str_t, _ = _pad_sequences(eb_str)
    eb_agt_t, _ = _pad_sequences(eb_agt)

    # ---- Pad atmosphere --------------------------------------------------
    atm_t, atm_mask = _pad_sequences(atm_ids_list)
    atm_cls_t, _ = _pad_sequences(atm_cls_list)
    atm_str_t, _ = _pad_sequences(atm_str_list)

    max_atm = atm_t.size(1)
    pos_arr = np.zeros((len(examples), max_atm, 2), dtype=np.float32)
    for i, positions in enumerate(atm_pos_list):
        for j, p in enumerate(positions[:max_atm]):
            pos_arr[i, j] = p

    return {
        "edge_a_ids":    ea_t.to(device),
        "edge_a_class":  ea_cls_t.to(device),
        "edge_a_stream": ea_str_t.to(device),
        "edge_a_agency": ea_agt_t.to(device),
        "edge_rel_ids":  er_t.to(device),
        "edge_b_ids":    eb_t.to(device),
        "edge_b_class":  eb_cls_t.to(device),
        "edge_b_stream": eb_str_t.to(device),
        "edge_b_agency": eb_agt_t.to(device),
        "edge_mask":     edge_mask.to(device),
        "atm_token_ids": atm_t.to(device),
        "atm_class":     atm_cls_t.to(device),
        "atm_stream":    atm_str_t.to(device),
        "atm_positions": torch.tensor(pos_arr, device=device),
        "atm_mask":      atm_mask.to(device),
        "cand_a_ids":    torch.tensor(ca_ids, dtype=torch.long, device=device),
        "cand_a_class":  torch.tensor(ca_cls, dtype=torch.long, device=device),
        "cand_a_stream": torch.tensor(ca_str, dtype=torch.long, device=device),
        "cand_a_agency": torch.tensor(ca_agt, dtype=torch.long, device=device),
        "cand_rel_ids":  torch.tensor(cr_ids_list, dtype=torch.long, device=device),
        "cand_b_ids":    torch.tensor(cb_ids, dtype=torch.long, device=device),
        "cand_b_class":  torch.tensor(cb_cls, dtype=torch.long, device=device),
        "cand_b_stream": torch.tensor(cb_str, dtype=torch.long, device=device),
        "cand_b_agency": torch.tensor(cb_agt, dtype=torch.long, device=device),
        "targets":       torch.tensor(targets, dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_connection_cartridge(
    spec_path: str,
    output_dir: str,
    n_paths: int = 1000,
    n_epochs: int = 50,
    device: str = "cpu",
) -> Tuple[MysteryConnectionModel, Dict[str, float]]:
    _ = output_dir  # handled by caller

    spec = CartridgeSpec.load(spec_path)
    model = MysteryConnectionModel(
        vocab_size=spec.vocab_size,
        embedding_dim=spec.embedding_dim,
        context_dim=spec.context_dim,
        n_attractor_dims=spec.n_attractor_dims,
        token_graph=spec.token_graph,
    ).to(device)

    id_to_idx, class_to_idx, stream_to_idx, agency_to_idx = _build_mappings(spec.tokens)

    print(f"\n[CONNECTION SAMPLING] requesting {n_paths} examples...", flush=True)
    sampler = ConnectionSampler(spec)
    positives = sampler.sample_batch(n_paths)

    if not positives:
        print("[CONNECTION TRAIN] ERROR: 0 positive examples — aborting.", flush=True)
        return model, {"loss": 0.0}

    negatives = [sampler.sample_negative(ex) for ex in positives]
    examples = positives + negatives
    np.random.shuffle(examples)

    print(
        f"[CONNECTION SAMPLING] done  total={len(examples)} "
        f"({len(positives)} pos / {len(negatives)} neg)",
        flush=True,
    )

    criterion = EdgeCoherenceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = min(64, len(examples))
    total_batches = max(1, (len(examples) + batch_size - 1) // batch_size)
    param_count = sum(p.numel() for p in model.parameters())

    print(
        f"\n[CONNECTION TRAIN] examples={len(examples)}  batch={batch_size}  "
        f"batches/epoch={total_batches}  params={param_count:,}",
        flush=True,
    )
    print("[CONNECTION TRAIN] epoch  loss     trend", flush=True)
    print("[CONNECTION TRAIN] " + "-" * 30, flush=True)

    history: Dict[str, float] = {"loss": 0.0}
    loss_history: List[float] = []

    for epoch in range(n_epochs):
        np.random.shuffle(examples)
        epoch_loss = 0.0

        for start in range(0, len(examples), batch_size):
            batch_ex = examples[start : start + batch_size]
            batch = _batchify_examples(
                batch_ex, id_to_idx, class_to_idx, stream_to_idx, agency_to_idx, device
            )

            model.train()
            out = model(
                batch["edge_a_ids"],   batch["edge_a_class"],
                batch["edge_a_stream"], batch["edge_a_agency"],
                batch["edge_rel_ids"],
                batch["edge_b_ids"],   batch["edge_b_class"],
                batch["edge_b_stream"], batch["edge_b_agency"],
                batch["edge_mask"],
                batch["atm_token_ids"], batch["atm_class"], batch["atm_stream"],
                batch["atm_positions"], batch["atm_mask"],
                batch["cand_a_ids"],   batch["cand_a_class"],
                batch["cand_a_stream"], batch["cand_a_agency"],
                batch["cand_rel_ids"],
                batch["cand_b_ids"],   batch["cand_b_class"],
                batch["cand_b_stream"], batch["cand_b_agency"],
            )

            loss = criterion(out["edge_coherence"], batch["targets"])

            if not torch.isfinite(loss):
                print(
                    f"[CONNECTION TRAIN] non-finite loss at epoch {epoch+1} — aborting.",
                    flush=True,
                )
                return model, {"loss": float("nan")}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())

        avg_loss = epoch_loss / total_batches
        history["loss"] = avg_loss
        loss_history.append(avg_loss)

        trend = ""
        if len(loss_history) >= 2:
            delta = loss_history[-1] - loss_history[-2]
            trend = f"{'↓' if delta < 0 else '↑'}{abs(delta):.4f}"

        print(
            f"[CONNECTION TRAIN] {epoch+1:02d}/{n_epochs}  {avg_loss:.4f}  {trend}",
            flush=True,
        )

    return model, history
