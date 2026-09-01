#!/usr/bin/env python3
"""Hotfix runner for open-ended RSNN growth selection.

Fixes the selection-phase crash caused by in-place writes to leaf Parameters
that require gradients. The permanent prune/reset writes are now performed
under torch.no_grad(), while all scoring/gradient calculations remain unchanged.
"""
from __future__ import annotations

import torch

import deepmind_rsnn_open_ended_three_way as base


class FixedOpenGrowthRSNN(base.OpenGrowthRSNN):
    def structural_step(self, optimizer, structure_bank: base.dm.Bank, criterion, *, seed: int):
        self.structural_cycle += 1
        event = {"cycle": self.structural_cycle, "phase": self.phase, "active_before": base.count_active(self)}
        shadow = self.shadow_scores(structure_bank, criterion)

        if self.phase == "growth":
            grown = self.grow_(optimizer, shadow, seed=seed)
            novelty = grown["total"] / base.TOTAL_WEIGHTS
            self.growth_streak = self.growth_streak + 1 if novelty < base.GROWTH_NOVELTY else 0
            event.update({"grown": grown, "growth_novelty": novelty, "growth_streak": self.growth_streak})
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
        grown = self.grow_(optimizer, shadow, seed=seed)
        contrib = self.contribution_scores(structure_bank, criterion)

        for n, _, mask in base.matrix_triplets(self):
            active = mask > 0.5
            vals = contrib[n][active]
            if vals.numel():
                k = max(1, int(round(vals.numel() * base.IMPORTANT_FRACTION)))
                thr = torch.topk(vals, k=k, largest=True).values.min()
                getattr(self, f"appearance_{n}")[active & (contrib[n] >= thr)] += 1

        combined, active_g, protected_g, spans = self._global_arrays(contrib)
        active_ids = torch.nonzero(active_g, as_tuple=False).flatten()
        kimp = max(1, int(round(active_ids.numel() * base.IMPORTANT_FRACTION)))
        important_ids = active_ids[torch.topk(combined[active_ids], k=kimp, largest=True).indices]
        important_set = set(int(x) for x in important_ids.cpu().tolist())
        novelty = 1.0 if self.prev_important is None else len(important_set - self.prev_important) / max(len(important_set), 1)
        self.prev_important = important_set
        self.selection_streak = self.selection_streak + 1 if novelty < base.SELECTION_NOVELTY else 0
        event.update({"grown": grown, "important_novelty": novelty, "selection_streak": self.selection_streak})

        if self.selection_streak >= base.SELECTION_STABLE_CYCLES:
            self.phase = "final"
            self.topology_stable = True
            event["transition"] = "selection_to_final"
            event["active_after"] = base.count_active(self)
            self.events.append(event)
            return event

        eligible = active_g & (~protected_g)
        eligible_ids = torch.nonzero(eligible, as_tuple=False).flatten()
        nprotect = min(max(1, int(round(active_ids.numel() * base.PROTECT_FRACTION))), int(eligible_ids.numel()))
        top_ids = eligible_ids[torch.topk(combined[eligible_ids], k=nprotect, largest=True).indices]
        self._apply_global_mask(top_ids, spans, "protected_{n}", True)

        _, active_g, protected_g, spans = self._global_arrays(contrib)
        removable = active_g & (~protected_g)
        rem_ids = torch.nonzero(removable, as_tuple=False).flatten()
        nprune = min(max(1, int(round(active_ids.numel() * base.PRUNE_FRACTION))), int(rem_ids.numel()))
        bottom_ids = rem_ids[torch.topk(combined[rem_ids], k=nprune, largest=False).indices]

        base_loss = float(base.bank_loss(self, structure_bank, criterion, require_grad=False).detach())
        saved_masks = {n: m.detach().clone() for n, _, m in base.matrix_triplets(self)}
        with torch.no_grad():
            self._apply_global_mask(top_ids, spans, "M_{n}", 0.0)
        top_loss = float(base.bank_loss(self, structure_bank, criterion, require_grad=False).detach())
        with torch.no_grad():
            for n, _, m in base.matrix_triplets(self):
                m.copy_(saved_masks[n])
            self._apply_global_mask(bottom_ids, spans, "M_{n}", 0.0)
        bottom_loss = float(base.bank_loss(self, structure_bank, criterion, require_grad=False).detach())
        with torch.no_grad():
            for n, _, m in base.matrix_triplets(self):
                m.copy_(saved_masks[n])

        # HOTFIX: all permanent Parameter/mask/optimizer-state mutations are no-grad.
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
            "protected_added": int(top_ids.numel()),
            "pruned": int(bottom_ids.numel()),
            "ablation_top_loss_delta": top_loss - base_loss,
            "ablation_bottom_loss_delta": bottom_loss - base_loss,
            "protected_total": sum(int(getattr(self, f"protected_{n}").sum()) for n, _, _ in base.matrix_triplets(self)),
            "active_after": base.count_active(self),
        })
        self.events.append(event)
        return event


# make_model() resolves OpenGrowthRSNN from the imported module's global namespace.
base.OpenGrowthRSNN = FixedOpenGrowthRSNN


if __name__ == "__main__":
    base.main()
