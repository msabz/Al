#!/usr/bin/env python3
"""Open-growth structural-selection implementation used by the long-run tests.

This version fixes two issues found in the previous experiment:
1) all permanent Parameter/mask/optimizer-state mutations are no-grad; and
2) the protected core is *rolling*, not cumulative. At every selection cycle,
   exactly the current global best 2% (dual persistence+contribution score)
   are protected from pruning. Protection is recomputed each cycle, so the
   entire network can never become protected merely because many cycles ran.

The worst 2% of the remaining active, unprotected synapses are pruned, given a
one-cycle cooldown, and may compete for regrowth later. This preserves the
user's intended loop: fill -> dual-score -> protect best 2% -> remove worst 2%
-> refill -> repeat until the important-set novelty is <1% for 3 cycles.
"""
from __future__ import annotations

import torch

import deepmind_rsnn_open_ended_three_way as base


class FixedOpenGrowthRSNN(base.OpenGrowthRSNN):
    def structural_step(self, optimizer, structure_bank: base.dm.Bank, criterion, *, seed: int):
        self.structural_cycle += 1
        event = {
            "cycle": self.structural_cycle,
            "phase": self.phase,
            "active_before": base.count_active(self),
        }
        shadow = self.shadow_scores(structure_bank, criterion)

        if self.phase == "growth":
            grown = self.grow_(optimizer, shadow, seed=seed)
            novelty = grown["total"] / base.TOTAL_WEIGHTS
            self.growth_streak = self.growth_streak + 1 if novelty < base.GROWTH_NOVELTY else 0
            event.update({
                "grown": grown,
                "growth_novelty": novelty,
                "growth_streak": self.growth_streak,
            })
            if self.growth_streak >= base.GROWTH_STABLE_CYCLES:
                self.phase = "selection"
                event["transition"] = "growth_to_selection"
            event["active_after"] = base.count_active(self)
            self.events.append(event)
            return event

        if self.phase != "selection":
            event["active_after"] = base.count_active(self)
            self.events.append(event)
            return event

        self.selection_cycles += 1

        # First refill every currently available dormant position that passes
        # the same shadow-gradient growth criterion.
        grown = self.grow_(optimizer, shadow, seed=seed)
        contrib = self.contribution_scores(structure_bank, criterion)

        # Persistence evidence: appearance in the top-20% contribution set.
        for n, _, mask in base.matrix_triplets(self):
            active = mask > 0.5
            vals = contrib[n][active]
            if vals.numel():
                k = max(1, int(round(vals.numel() * base.IMPORTANT_FRACTION)))
                thr = torch.topk(vals, k=k, largest=True).values.min()
                getattr(self, f"appearance_{n}")[active & (contrib[n] >= thr)] += 1

        # Combined dual score = persistence x current contribution.
        combined, active_g, _, spans = self._global_arrays(contrib)
        active_ids = torch.nonzero(active_g, as_tuple=False).flatten()
        if active_ids.numel() == 0:
            raise RuntimeError("open-growth selection has no active synapses")

        kimp = max(1, int(round(active_ids.numel() * base.IMPORTANT_FRACTION)))
        important_ids = active_ids[
            torch.topk(combined[active_ids], k=kimp, largest=True).indices
        ]
        important_set = set(int(x) for x in important_ids.cpu().tolist())
        novelty = (
            1.0
            if self.prev_important is None
            else len(important_set - self.prev_important) / max(len(important_set), 1)
        )
        self.prev_important = important_set
        self.selection_streak = (
            self.selection_streak + 1 if novelty < base.SELECTION_NOVELTY else 0
        )
        event.update({
            "grown": grown,
            "important_novelty": novelty,
            "selection_streak": self.selection_streak,
        })

        # When the important set changes by <1% for three consecutive cycles,
        # freeze the final topology and continue ordinary weight training.
        if self.selection_streak >= base.SELECTION_STABLE_CYCLES:
            self.phase = "final"
            self.topology_stable = True
            event["transition"] = "selection_to_final"
            event["active_after"] = base.count_active(self)
            event["protected_total"] = sum(
                int(getattr(self, f"protected_{n}").sum())
                for n, _, _ in base.matrix_triplets(self)
            )
            self.events.append(event)
            return event

        # ROLLING CORE FIX:
        # Protect the current global best 2% of *all active* synapses. Do not
        # accumulate protection forever. A synapse remains protected only while
        # its dual score keeps it in the current best 2%.
        nprotect = max(1, int(round(active_ids.numel() * base.PROTECT_FRACTION)))
        nprotect = min(nprotect, int(active_ids.numel()))
        top_ids = active_ids[
            torch.topk(combined[active_ids], k=nprotect, largest=True).indices
        ]
        with torch.no_grad():
            for n, _, _ in base.matrix_triplets(self):
                getattr(self, f"protected_{n}").zero_()
            self._apply_global_mask(top_ids, spans, "protected_{n}", True)

        # Rebuild global protection view after replacing the rolling core.
        _, active_g, protected_g, spans = self._global_arrays(contrib)
        removable = active_g & (~protected_g)
        rem_ids = torch.nonzero(removable, as_tuple=False).flatten()
        nprune = max(1, int(round(active_ids.numel() * base.PRUNE_FRACTION)))
        nprune = min(nprune, int(rem_ids.numel()))
        bottom_ids = rem_ids[
            torch.topk(combined[rem_ids], k=nprune, largest=False).indices
        ] if nprune > 0 else rem_ids

        # Group ablations are diagnostics only; they do not alter training state.
        base_loss = float(
            base.bank_loss(self, structure_bank, criterion, require_grad=False).detach()
        )
        saved_masks = {n: m.detach().clone() for n, _, m in base.matrix_triplets(self)}
        with torch.no_grad():
            self._apply_global_mask(top_ids, spans, "M_{n}", 0.0)
        top_loss = float(
            base.bank_loss(self, structure_bank, criterion, require_grad=False).detach()
        )
        with torch.no_grad():
            for n, _, m in base.matrix_triplets(self):
                m.copy_(saved_masks[n])
            if bottom_ids.numel():
                self._apply_global_mask(bottom_ids, spans, "M_{n}", 0.0)
        bottom_loss = float(
            base.bank_loss(self, structure_bank, criterion, require_grad=False).detach()
        )
        with torch.no_grad():
            for n, _, m in base.matrix_triplets(self):
                m.copy_(saved_masks[n])

        # All permanent Parameter/mask/optimizer-state mutations are no-grad.
        with torch.no_grad():
            weights = {a: b for a, b, _ in base.matrix_triplets(self)}
            masks = {a: c for a, _, c in base.matrix_triplets(self)}
            for n, start, end in spans:
                local = bottom_ids[(bottom_ids >= start) & (bottom_ids < end)] - start
                if local.numel() == 0:
                    continue
                w = weights[n]
                mask = masks[n]
                changed = torch.zeros_like(mask, dtype=torch.bool).view(-1)
                changed[local] = True
                changed = changed.view_as(mask)
                mask.view(-1)[local] = 0.0
                w.view(-1)[local] = 0.0
                getattr(self, f"cooldown_{n}").view(-1)[local] = 1
                getattr(self, f"utility_{n}").view(-1)[local] = 0.0
                base.zero_optimizer_positions_(optimizer, w, changed)
            self.apply_masks_()

        event.update({
            "rolling_core": True,
            "protected_target": nprotect,
            "protected_total": sum(
                int(getattr(self, f"protected_{n}").sum())
                for n, _, _ in base.matrix_triplets(self)
            ),
            "pruned": int(bottom_ids.numel()),
            "ablation_top_loss_delta": top_loss - base_loss,
            "ablation_bottom_loss_delta": bottom_loss - base_loss,
            "active_after": base.count_active(self),
        })
        self.events.append(event)
        return event


# make_model() resolves OpenGrowthRSNN from the imported module's global namespace.
base.OpenGrowthRSNN = FixedOpenGrowthRSNN


if __name__ == "__main__":
    base.main()
