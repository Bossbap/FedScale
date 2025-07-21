import math, time, torch, numpy as np
from overrides import overrides
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.dataloaders.nlp import mask_tokens
from typing import List, Dict
import logging
import traceback
import time

# ---------------------------------------------------------------------- #
#  Simple helpers using client-trace arrays                              #
# ---------------------------------------------------------------------- #
def piecewise_lookup(ts, vals, t):
    norm = t % (48 * 3600)
    if norm < ts[0]:
        return vals[0]
    idx = np.searchsorted(ts, norm, side='right') - 1
    return vals[idx]

def bandwidth(trace, t):
    rate = piecewise_lookup(trace['timestamps_livelab'], trace['rate'], t)
    return trace['peak_throughput'] * rate / 54.0        # Mb s⁻¹

def compute_speed(trace, t, round_noise):
    peak = trace['cpu_flops'] + trace['gpu_flops']
    avail = piecewise_lookup(trace['timestamps_carat'],
                             trace['availability'], t) / 100.0
    batt  = piecewise_lookup(trace['timestamps_carat'],
                             trace['batteryLevel'], t)
    batt_factor = 1.0 if batt >= 70 else (0.9 if batt >= 50 else
                   (0.8 if batt >= 30 else (0.6 if batt >= 10 else 0.4)))
    return peak * avail * batt_factor * round_noise

def compress_topk(delta, frac):
    if frac >= 1.0: return [d.clone() for d in delta]
    flat = torch.cat([d.view(-1).abs() for d in delta])
    k = max(1, int(flat.numel() * frac))
    thresh = torch.kthvalue(flat, flat.numel() - k + 1).values.item()
    return [d * (d.abs() >= thresh) for d in delta]

def dense_packet(tensors):
    return {f"param_{i}": t.cpu().numpy() for i, t in enumerate(tensors)}

# ---------------------------------------------------------------------- #
class AdaptiveTorchClient(TorchClient):

    def __init__(self, args):
        super().__init__(args)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _l2_norm(tensors: List[torch.Tensor]) -> float:
        return torch.sqrt(sum((t.float() ** 2).sum() for t in tensors)).item()

    @staticmethod
    def _simulate_compute(trace: Dict, start_t: float,
                      steps: int, flops_step: float,
                      round_noise: float, clock: int) -> float:
        WINDOW = 48 * 3600
        rem_ops = flops_step * clock * steps
        t       = start_t

        t0_real = time.time()
        warned  = False    

        EPS = 1e-9
        while rem_ops > EPS:
            speed = compute_speed(trace, t, round_noise)
            speed = max(speed, 1e-6)

            ts   = trace['timestamps_carat']
            idx  = np.searchsorted(ts, (t % WINDOW), side='right')
            nxt  = t - (t % WINDOW) + (ts[idx] if idx < len(ts) else WINDOW)
            dt   = min(rem_ops / speed, nxt - t)

            # ---- 60‑second watchdog --------------------------------------- ▲
            if not warned and time.time() - t0_real > 60:
                logging.warning("[simulate_compute] cid=?  "
                                "t=%.2f  rem_ops=%.2e  speed=%.2e  "
                                "idx=%d  ts_idx=%.2f  nxt=%.2f  dt=%.2f",
                                t, rem_ops, speed, idx,
                                ts[idx] if idx < len(ts) else -1, nxt, dt)
                warned = True                                                  
            # ---------------------------------------------------------------- ▼

            rem_ops -= speed * dt
            t       += dt

        return t - start_t


    @staticmethod
    def sample_round_noise(target_mean: float = 0.9, sigma: float = 0.25):
        """One log-normal noise multiplier per round (σ from original code)."""
        # Noise
        mu = np.log(target_mean) - (sigma**2)/2
        return np.random.lognormal(mean=mu, sigma=sigma)

    # ------- one SGD step (unchanged except signature) ----------------- #
    def _one_step(self, batch, conf, model, optim, crit, tokenizer):
        if conf.task == 'nlp':
            (data, _) = batch
            data, target = mask_tokens(data, tokenizer, conf, device=self.device)
        elif conf.task == 'voice':
            (data, target, ipct, t_sizes), _ = batch
            ipct = ipct.mul_(int(data.size(3))).int()
        elif conf.task == 'detection':
            tmp = batch; target = tmp[4]; data = tmp[0:4]
        else:
            (data, target) = batch

        if conf.task == "detection":
            self.im_data.resize_(data[0].size()).copy_(data[0])
            self.im_info.resize_(data[1].size()).copy_(data[1])
            self.gt_boxes.resize_(data[2].size()).copy_(data[2])
            self.num_boxes.resize_(data[3].size()).copy_(data[3])
        elif conf.task == 'speech':
            data = torch.unsqueeze(data, 1).to(self.device)
        elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
            (data, masks) = data
            data, masks = data.to(self.device), masks.to(self.device)
        else:
            data = data.to(self.device)
        target = target.to(self.device)

        if conf.task == 'nlp':
            loss = model(data, labels=target)[0]
        elif conf.task == 'voice':
            out, out_sz = model(data, ipct)
            out = out.transpose(0,1).float()
            loss = crit(out, target, out_sz, t_sizes)
        elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
            loss = model(data, attention_mask=masks, labels=target).loss
        elif conf.task == 'detection':
            *_, rpn_c, rpn_b, rcnn_c, rcnn_b, _ = \
                model(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)
            loss = rpn_c + rpn_b + rcnn_c + rcnn_b
        else:
            loss = crit(model(data), target)

        # ensure scalar loss for autograd
        if loss.dim() != 0:                 # e.g. reduction='none'
            loss_vec  = loss.detach()
            loss_mean = loss_vec.mean()
        else:                               # already scalar
            loss_vec  = loss.detach().view(1)
            loss_mean = loss_vec[0]


        optim.zero_grad()
        loss_mean.backward()
        optim.step()
        return loss_mean.item(), loss_vec.cpu().tolist()

    # ----------------------------- train -------------------------------- #
    @overrides
    def train(self, client_data, model, conf):
        try:
            # ----------- initialise round‑specific state -------------------
            trace         = conf.dynamic_trace
            tokenizer     = getattr(conf, "tokenizer", None)
            model         = model.to(self.device).train()
            optim         = self.get_optimizer(model, conf)
            crit          = self.get_criterion(conf)
    
            round_noise   = self.sample_round_noise(target_mean = 0.9, sigma = 0.25)

            flops_step    = conf.model_amount_parameters * conf.batch_size * 3.0
            recheck       = max(1, conf.budget_recheck_steps)          # “E”
            reduction     = 0.5                                        # hard‑coded
            min_frac      = conf.min_payload_frac
            clock         = conf.clock_factor

    
            curr_t        = conf.start_time            # includes download
            budget        = conf.t_budget_train        # = t_budget – t_download
            keep_frac     = 1.0                        # start with full payload
    
            steps_done    = 0
            self.loss_sq_sum  = 0.0
            self.seen_samples = 0
            ewma          = 0.0
            λ             = conf.ewma_lambda
    
            # snapshot of initial weights for delta calculation
            w0 = [p.data.clone() for p in model.parameters()]
            data_it = iter(client_data)

            # ---------------- LOG BUFFERS ---------------------------------
            est_iters_hist, est_tcomm_hist, speed_hist, bw_hist = [], [], [], []   # phase‑2
            time_hist_2,   budget_hist_2   = [], []                                # NEW
            gain_comm_hist = []                                                    # phase‑3
            time_hist_3,   budget_hist_3   = [], []                                # NEW
            speed_hist_3,  bw_hist_3       = [], []                                # NEW
    
            # ----------------- PHASE 2 : coarse fit‑as‑many ----------------
            while True:

                speed_now   = compute_speed(trace, curr_t, round_noise)
                t_comp_est  = (flops_step * clock) / max(speed_now, 1e-6)
                up_rate_now = bandwidth(trace, curr_t)
                t_comm_est  = (conf.model_size * clock * keep_frac / reduction) / max(up_rate_now, 1e-6)
                
                if budget <= t_comm_est:
                    break
    
                max_iters   = int((budget - t_comm_est) // t_comp_est)

                # ---- log the estimate -----------------------------------
                est_iters_hist.append(max_iters)
                est_tcomm_hist.append(t_comm_est)
                speed_hist.append(speed_now)
                bw_hist.append(up_rate_now)
                time_hist_2.append(curr_t)
                budget_hist_2.append(budget)

                if max_iters < recheck:
                    break                            # switch to phase 3
    
                # ---- run exactly `recheck` iterations, simulate real time -
                elapsed = self._simulate_compute(trace, curr_t, recheck,
                                                flops_step, round_noise, clock)

                curr_t  += elapsed
                budget  -= elapsed
    
                for i in range(recheck):
                    try:
                        batch = next(data_it)
                    except StopIteration:
                        data_it = iter(client_data)
                        batch = next(data_it)
                    
    
                    loss_val, loss_list = self._one_step(
                        batch, conf, model, optim, crit, tokenizer)
                    self.loss_sq_sum  += sum(l**2 for l in loss_list)
                    self.seen_samples += len(loss_list)
    
                    # --- loss‑tracking identical to TorchClient ----------
                    if steps_done < len(client_data):
                        if self.epoch_train_loss == 1e-4:
                            self.epoch_train_loss = loss_val
                        else:
                            self.epoch_train_loss = \
                                (1-conf.loss_decay)*self.epoch_train_loss   +\
                                conf.loss_decay * loss_val
    
                    steps_done        += 1
                    self.completed_steps = steps_done
    
                # (exact‑equal case): exit now if we just consumed the last chunk
                if max_iters == recheck:
                    break
    
            # ----------------- PHASE 3 : fine trade‑off --------------------
            prev_comp_norm = 0.0
            steps_done_bis = 0
    
            while True:
                # ---- one SGD step with real‑time simulation --------------                        
                elapsed = self._simulate_compute(trace, curr_t, 1,
                                            flops_step, round_noise, clock)
                
                curr_t  += elapsed
                budget  -= elapsed

                time_hist_3.append(curr_t)
                budget_hist_3.append(budget)
                speed_hist_3.append(speed_now)       # reuse speed_now you already compute
                bw_hist_3.append(up_rate_now)
    
                try:
                    batch = next(data_it)
                except StopIteration:
                    data_it = iter(client_data)
                    batch = next(data_it)
    
                loss_val, loss_list = self._one_step(
                    batch, conf, model, optim, crit, tokenizer)
                self.loss_sq_sum  += sum(l**2 for l in loss_list)
                self.seen_samples += len(loss_list)
                    
                steps_done        += 1
                steps_done_bis        += 1
                self.completed_steps = steps_done
    
                # loss EWMA for logging (same as above)
                if steps_done <= len(client_data):
                    if self.epoch_train_loss == 1e-4:
                        self.epoch_train_loss = loss_val
                    else:
                        self.epoch_train_loss = \
                            (1-conf.loss_decay)*self.epoch_train_loss   +\
                            conf.loss_decay * loss_val
    
                # ---- delta   norms ---------------------------------------
                delta       = [p.data - w0i for p, w0i in zip(model.parameters(), w0)]
                full_norm   = self._l2_norm(delta)
    
                up_rate_now = bandwidth(trace, curr_t)
                t_comm_full = (conf.model_size * clock / reduction) / max(up_rate_now, 1e-6)
                keep_frac   = max(min_frac,
                                min(1.0, budget / t_comm_full))
                
                comp_delta  = compress_topk(delta, keep_frac)    # Cₙ₊¹(Δₙ₊¹)
                comp_norm   = self._l2_norm(comp_delta)          # = Uₙ₊¹

                gain_comm   = comp_norm - prev_comp_norm         # Uₙ₊¹ − Uₙ
                # Optional EWMA smoothing
                # ewma = λ * ewma + (1-λ) * gain_comm
                # gain_comm = ewma

                if gain_comm <= 0 or keep_frac <= min_frac or budget <= 0:
                    break

                gain_comm_hist.append(gain_comm)
                prev_comp_norm = comp_norm
    
            # -------------------- upload simulation -----------------------
            upload_time = ClientUploadSimulator.simulate(
                trace, curr_t, conf.model_size * clock * keep_frac,
                reduction_factor=reduction,
            )
    
            total_wall  = conf.t_download + (curr_t - conf.start_time) + upload_time
    
            # ------------- build compressed update for aggregator ---------
            final_delta = compress_topk(
                [p.data - w0i for p, w0i in zip(model.parameters(), w0)],
                max(keep_frac, min_frac)
            )
            full_sd = model.state_dict()
            k_iter  = iter(final_delta)                      # same order as model.parameters()
            for name, tensor in full_sd.items():
                if tensor.requires_grad:                    # parameter → maybe compressed
                    tensor.copy_(next(k_iter))              # already zero‑masked
                # else: buffer (running_mean, etc.) → leave unchanged

            update_dict = {
                f"param_{i}": t.cpu().numpy()
                for i, t in enumerate(full_sd.values())
            }

            trained_unique = min(len(client_data.dataset),
                                steps_done * conf.batch_size)

            # ----------- utility: same definition as vanilla --------------
            rms_loss    = math.sqrt(self.loss_sq_sum / max(1, self.seen_samples))
            utility_val = rms_loss * float(trained_unique)   # |Bᵢ|·RMS


            return {
                "client_id"    : conf.client_id,
                "moving_loss"  : self.epoch_train_loss,
                "trained_size" : trained_unique,
                "success"      : True,
                "utility"      : utility_val,
                "update_weight": update_dict,
                "wall_duration": total_wall,
            }
        except Exception:                      # <-- NEW
            # Full stack‑trace into the executor log
            logging.error("[adaptive_torch_client] EXCEPTION on client %s\n%s",
                          conf.client_id, traceback.format_exc())

            # Minimal failure packet so the executor can still report back
            return {
                "client_id"    : conf.client_id,
                "moving_loss"  : 0.0,
                "trained_size" : 0,
                "success"      : False,
                "utility"      : 0.0,
                "update_weight": {},      # empty so payload is tiny
                "wall_duration": 0.0,
            }


# ---------------------------------------------------------------------- #
#  Simple upload simulator re-using _simulate_data_phase logic ---------- #
# ---------------------------------------------------------------------- #
class ClientUploadSimulator:
    @staticmethod
    def _simulate_phase(start_time, work_mb, trace, reduction=1.0):
        WINDOW = 48 * 3600
        t = start_time
        rem = work_mb / reduction
        while True:
            bw = bandwidth(trace, t)
            if bw <= 1e-6: raise RuntimeError("Zero BW")
            # next breakpoint
            ts = trace['timestamps_livelab']
            idx = np.searchsorted(ts, (t % WINDOW), side='right')
            nxt = t - (t % WINDOW) + (ts[idx] if idx < len(ts) else WINDOW)
            dt  = min(rem / bw, nxt - t)
            if rem <= bw * dt + 1e-9:
                return t + dt
            rem -= bw * dt; t += dt

    @classmethod
    def simulate(cls, trace, start_time, payload_mb, reduction_factor=0.5):
        end = cls._simulate_phase(start_time, payload_mb,
                                  trace, reduction_factor)
        return end - start_time