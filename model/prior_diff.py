import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import PredefinedNoiseScheduleDiscrete, get_entropy, fuse_logits_by_log_probs, sin_mask_ratio_adapter
from model.diffusion import DiscreteUniformTransition, BlosumTransition, DiscreteMarginalTransition


class Prior_Diff(nn.Module):
    def __init__(self, model, prior_model, timesteps=500, loss_type='CE', objective='pred_x0', noise_type='marginal',
                 sample_method='ddim', min_mask_ratio=0.4, dev_mask_ratio=0.1, ensemble_num=50,
                 marginal_dist_path='data/train_magrinal_x.pt'):
        super().__init__()
        self.model = model
        self.prior_model = prior_model
        self.objective = objective
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.noise_type = noise_type
        self.sample_method = sample_method
        self.min_mask_ratio = min_mask_ratio
        self.dev_mask_ratio = dev_mask_ratio
        self.ensemble_num = ensemble_num
        if noise_type == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=20)
        elif noise_type == 'blosum':
            self.transition_model = BlosumTransition(timestep=self.timesteps + 1)
        elif noise_type == 'marginal':
            self.transition_model = DiscreteMarginalTransition(x_classes=20, x_marginal_path=marginal_dist_path)

        assert objective in {'pred_noise', 'pred_x0'}

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule='cosine', timesteps=self.timesteps)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'CE':
            return F.cross_entropy

    def apply_noise(self, data, t_int):
        t_float = t_int / self.timesteps
        if self.noise_type == 'uniform' or self.noise_type == 'marginal':
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        else:
            Qtb = self.transition_model.get_Qt_bar(t_float, device=data.x.device)
            alpha_t_bar = None
        # prob_X = (Qtb[data.batch] @ data.x[:, :20].unsqueeze(2)).squeeze()
        prob_X = (data.x[:, :20].unsqueeze(1) @ Qtb[data.batch]).squeeze()
        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t, num_classes=20)
        noise_data = data.clone()
        noise_data.x = noise_X
        return noise_data, alpha_t_bar

    def sample_discrete_feature_noise(self, limit_dist, num_node):
        x_limit = limit_dist[None, :].expand(num_node, -1)  # [num_node,20]
        U_X = x_limit.flatten(end_dim=-2).multinomial(1).squeeze()
        U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
        return U_X

    def diffusion_loss(self, data, t_int):
        '''
        Compute the divergence between  q(x_t-1|x_t,x_0) and p_{\theta}(x_t-1|x_t)
        '''
        # q(x_t-1|x_t,x_0)
        s_int = t_int - 1
        t_float = t_int / self.timesteps
        s_float = s_int / self.timesteps
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=data.x.device)
        Qt = self.transition_model.get_Qt(beta_t, data.x.device)
        # prob_X = (Qtb[data.batch] @ data.x[:, :20].unsqueeze(2)).squeeze()
        prob_X = (data.x[:, :20].unsqueeze(1) @ Qtb[data.batch]).squeeze()
        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t, num_classes=20).type_as(data.x)
        prob_true = self.compute_posterior_distribution(noise_X, Qt, Qsb, Qtb, data)  # [N,d_t-1]

        # p_{\theta}(x_t-1|x_t) = \sum_{x0} q(x_t-1|x_t,x_0)p(x0|xt)
        noise_data = data.clone()
        noise_data.x = noise_X  # x_t
        t = t_int * torch.ones(size=(data.batch[-1] + 1, 1), device=data.x.device).float()
        pred = self.model(noise_data, t)
        pred_X = F.softmax(pred, dim=-1)  # \hat{p(X)}_0
        p_s_and_t_given_0_X = self.compute_batched_over0_posterior_distribution(X_t=noise_X, Q_t=Qt, Qsb=Qsb, Qtb=Qtb,
                                                                                data=data)  # [N,d0,d_t-1] 20,20
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # [N,d0,d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=1)  # [N,d_t-1]
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_pred = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # [N,d_t-1]
        loss = self.loss_fn(prob_pred, prob_true, reduction='mean')
        return loss

    def compute_val_loss(self, data):
        '''
        Compute the divergence between  q(x_t-1|x_t,x_0) and p_{\theta}(x_t-1|x_t)
        '''
        t_int = torch.randint(0, self.timesteps + 1, size=(data.batch[-1] + 1, 1), device=data.x.device).float()
        diffusion_loss = self.diffusion_loss(data, t_int)
        return diffusion_loss

    def compute_batched_over0_posterior_distribution(self, X_t, Q_t, Qsb, Qtb, data):
        """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
        """
        # X_t is a sample of q(x_t|x_t+1)
        Qt_T = Q_t.transpose(-1, -2)
        X_t_ = X_t.unsqueeze(dim=-2)
        left_term = X_t_ @ Qt_T[data.batch]  # [N,1,d_t-1]
        # left_term = left_term.unsqueeze(dim = 1) #[N,1,dt-1]

        right_term = Qsb[data.batch]  # [N,d0,d_t-1]

        numerator = left_term * right_term  # [N,d0,d_t-1]

        prod = Qtb[data.batch] @ X_t.unsqueeze(dim=2)  # N,d0,1
        denominator = prod
        denominator[denominator == 0] = 1e-6

        out = numerator / denominator

        return out

    def compute_posterior_distribution(self, M_t, Qt_M, Qsb_M, Qtb_M, data):
        """
        M_t: X_t
        Compute  q(x_t-1|x_t,x_0) = xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        """

        # X_t is a sample of q(x_t|x_t+1)
        Qt_T = Qt_M.transpose(-1, -2)
        X_t = M_t.unsqueeze(dim=-2)
        left_term = X_t @ Qt_T[data.batch]  # [N,1,d_t-1]

        M_0 = data.x.unsqueeze(dim=-2)  # [N,1,d_t-1]
        right_term = M_0 @ Qsb_M[data.batch]  # [N,1,dt-1]
        numerator = (left_term * right_term).squeeze()  # [N,d_t-1]

        X_t_T = M_t.unsqueeze(dim=-1)
        prod = M_0 @ Qtb_M[data.batch] @ X_t_T  # [N,1,1]
        denominator = prod.squeeze()
        denominator[denominator == 0] = 1e-6

        out = (numerator / denominator.unsqueeze(dim=-1)).squeeze()

        return out  # [N,d_t-1]

    def sample_p_zs_given_zt(self, t, s, zt, g_data, ipa_data, cond, diverse, sample_type, last_step):
        """
        sample zs~p(zs|zt)
        """
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        if self.noise_type == 'uniform' or self.noise_type == 'marginal':
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, g_data.x.device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, g_data.x.device)
        else:
            Qtb = self.transition_model.get_Qt_bar(t, g_data.x.device)
            Qsb = self.transition_model.get_Qt_bar(s, g_data.x.device)

        if sample_type == 'ddpm':
            Qt = self.transition_model.get_Qt(beta_t, g_data.x.device)
        elif sample_type == 'ddim':
            Qt = (Qsb / Qtb) / (Qsb / Qtb).sum(dim=-1).unsqueeze(dim=2)  # approximate
        else:
            raise NotImplementedError

        noise_data = g_data.clone()
        noise_data.x = zt

        ipa_noise_data = ipa_data.clone()
        base_logits = self.model(noise_data, t * self.timesteps)
        log_probs = F.log_softmax(base_logits, dim=-1)
        base_pred_x = log_probs.argmax(dim=-1)
        base_pred_x = F.one_hot(base_pred_x, num_classes=20).float()

        entropy = get_entropy(log_probs)
        mask_entropy = torch.zeros_like(entropy, dtype=torch.bool)
        unique_batches = noise_data.batch.unique()

        mask_ratios = sin_mask_ratio_adapter(1 - alpha_t_bar, max_deviation=self.dev_mask_ratio, center=self.min_mask_ratio)
        for mask_ratio, b in zip(mask_ratios, unique_batches):
            mask_entropy[noise_data.batch == b] = entropy[noise_data.batch == b] > torch.quantile(
                entropy[noise_data.batch == b], 1 - mask_ratio)

        ipa_noise_data.x_mask[ipa_noise_data.x_pad == 1] = mask_entropy.long()
        ipa_noise_data.x[ipa_noise_data.x_pad == 1] = base_pred_x

        prior_logits = self.prior_model(ipa_noise_data.x, ipa_noise_data.atom_pos, ipa_noise_data.x_mask,
                                        ipa_noise_data.x_pad)
        prior_log_probs = F.log_softmax(prior_logits, dim=-1)

        prior_logits = prior_logits[ipa_noise_data.x_pad == 1]
        prior_log_probs = prior_log_probs[ipa_noise_data.x_pad == 1]

        # fuse log probs
        logits = fuse_logits_by_log_probs([log_probs, prior_log_probs], [base_logits, prior_logits])
        # logits = prior_logits
        pred_X = F.softmax(logits, dim=-1)

        if last_step:
            sample_s = pred_X.argmax(dim=1)
            final_predicted_X = F.one_hot(sample_s, num_classes=20).float()
            return logits, final_predicted_X

        p_s_and_t_given_0_X = self.compute_batched_over0_posterior_distribution(X_t=zt, Q_t=Qt, Qsb=Qsb, Qtb=Qtb,
                                                                                data=g_data)  # [N,d0,d_t-1] 20,20 approximate Q_t-s with Qt
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # [N,d0,d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=1)  # [N,d_t-1]
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # [N,d_t-1]
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        if diverse:
            sample_s = prob_X.multinomial(1).squeeze()
        else:
            sample_s = prob_X.argmax(dim=1).squeeze()

        predicted_X_s = F.one_hot(sample_s, num_classes=20).float()

        return None, predicted_X_s

    def mc_ddim_sample(self, g_data, ipa_data=None, cond=False, diverse=True, stop=0, step=50):
        if self.noise_type == 'uniform' or self.noise_type == 'blosum':
            limit_dist = torch.ones(20) / 20
        elif self.noise_type == 'marginal':
            limit_dist = self.transition_model.x_marginal
        zt = self.sample_discrete_feature_noise(limit_dist=limit_dist, num_node=g_data.x.shape[0])  # [N,20] one hot
        zt = zt.to(g_data.x.device)
        # for s_int in tqdm(list(reversed(range(stop, self.timesteps, step)))):  500
        for s_int in reversed(range(stop, self.timesteps, step)):  # 500
            # z_t-1 ~p(z_t-1|z_t),
            s_array = s_int * torch.ones((g_data.batch[-1] + 1, 1)).type_as(g_data.x)
            t_array = s_array + step
            s_norm = s_array / self.timesteps
            t_norm = t_array / self.timesteps
            logits, zt = self.sample_p_zs_given_zt(t_norm, s_norm, zt, g_data, ipa_data, cond, diverse, self.sample_method,
                                                   last_step=s_int == 0)
        return logits, zt

    def forward(self, g_data, ipa_data=None):
        t_int = torch.randint(0, self.timesteps + 1, size=(g_data.batch[-1] + 1, 1), device=g_data.x.device).float()
        noise_data, alpha_t_bar = self.apply_noise(g_data, t_int)

        if self.objective == 'pred_x0':
            target = g_data.x
        else:
            raise ValueError(f'unknown objective {self.objective}')

        base_logits = self.model(noise_data, t_int)
        log_probs = F.log_softmax(base_logits, dim=-1)
        base_pred_x = log_probs.argmax(dim=-1)
        # one-hot encoding base_pred_x
        base_pred_x = F.one_hot(base_pred_x, num_classes=20).float()
        entropy = get_entropy(log_probs)
        mask_entropy = torch.zeros_like(entropy, dtype=torch.bool)
        unique_batches = noise_data.batch.unique()

        mask_ratios = sin_mask_ratio_adapter(1 - alpha_t_bar, max_deviation=self.dev_mask_ratio, center=self.min_mask_ratio)
        for mask_ratio, b in zip(mask_ratios, unique_batches):
            mask_entropy[noise_data.batch == b] = entropy[noise_data.batch == b] > torch.quantile(
                entropy[noise_data.batch == b], 1 - mask_ratio)

        ipa_data.x_mask[ipa_data.x_pad == 1] = mask_entropy.long()
        ipa_data.x[ipa_data.x_pad == 1] = base_pred_x

        prior_logits = self.prior_model(ipa_data.x, ipa_data.atom_pos, ipa_data.x_mask, ipa_data.x_pad)
        base_loss = self.loss_fn(base_logits, target, reduction='mean')
        mask_loss = self.loss_fn(prior_logits[ipa_data.x_mask == 1], ipa_data.label[ipa_data.x_mask == 1], reduction='mean')

        return base_loss, mask_loss
