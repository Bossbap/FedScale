class AdaptiveTorchClient(TorchClient):
    def train(self, client_data, model, conf):
        device = self.device
        model  = model.to(device).train()

        # --- 0. book-keeping -------------------------------------------------
        w0        = [p.data.clone() for p in model.parameters()]
        step_gain = 0.0
        ewma      = 0.0
        keep_frac = 1.0                       # Pₙ starts at 100 %
        comp_time_left = conf.t_budget        # wall-clock budget in *seconds*

        # crude per-iteration compute cost (sec); used only for budgeting
        compute_per_step = comp_time_left / conf.local_steps

        # --- 1. training loop ----------------------------------------------
        optim  = self.get_optimizer(model, conf)
        crit   = self.get_criterion(conf)

        for batch_idx, batch in enumerate(client_data):

            # stop if budget exhausted
            if comp_time_left < compute_per_step:
                break

            # ------- forward / backward / step -----------------------------
            self.train_step([batch], conf, model, optim, crit)

            # remaining budget
            comp_time_left -= compute_per_step

            # every budget_recheck_steps, evaluate gain vs loss -------------
            if (batch_idx + 1) % conf.budget_recheck_steps == 0:
                # Δ after current step
                w_curr = [p.data for p in model.parameters()]
                delta  = [(wc - w0i) for wc, w0i in zip(w_curr, w0)]

                full_norm = torch.sqrt(sum((d**2).sum() for d in delta)).item()

                # magnitude-based compression
                compressed = compress_topk(delta, keep_frac)
                comp_norm  = torch.sqrt(sum((d**2).sum() for d in compressed)).item()

                delta_gain = full_norm - step_gain   # step_gain holds previous norm
                delta_loss = full_norm - comp_norm
                step_gain  = full_norm

                score      = delta_gain - delta_loss
                ewma       = conf.ewma_lambda * ewma + (1-conf.ewma_lambda) * score

                if ewma <= 0 or keep_frac < conf.min_payload_frac:
                    break          # stop local SGD

                # update keep_frac toward smaller payload if budget tightens
                # (simple heuristic; can be refined later)
                keep_frac = max(conf.min_payload_frac,
                                comp_time_left / conf.t_budget)

        # --- 2. build dense update to return -------------------------------
        update_dense = pad_sparse_to_dense(compress_topk(
            [(p.data - w0i) for p, w0i in zip(model.parameters(), w0)],
            keep_frac
        ))

        results = {
            "client_id"     : conf.client_id,
            "update_weight" : update_dense,
            "moving_loss"   : self.epoch_train_loss,
            "trained_size"  : (batch_idx+1) * conf.batch_size,
            "success"       : True,
            "utility"       : step_gain,
            "wall_duration" : 0,     # still simulated by server
        }
        return results