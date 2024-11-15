import torch
import torch.nn.functional as F


class DiscreteUniformTransition:
    def __init__(self, x_classes: int):
        self.X_classes = x_classes

        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx)
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)

        return q_x

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx)
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x

        return q_x


class DiscreteMarginalTransition:
    def __init__(self, x_classes=20, x_marginal_path='data/train_magrinal_x.pt'):
        self.X_classes = x_classes
        self.x_marginal = torch.load(x_marginal_path)
        self.u_x = self.x_marginal.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)

    def get_Qt(self, beta_t, device):
        """
        Returns one-step transition matrices for X from step t - 1 to step t.
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        return q_x

    def get_Qt_bar(self, alpha_bar_t, device):
        """
        Returns t-step transition matrices for X from step 0 to step t.
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        return q_x


class BlosumTransition:
    def __init__(self, blosum_path='data/blosum_substitute.pt', x_classes=20, timestep=500):
        try:
            self.original_score, self.temperature_list, self.Qt_temperature = torch.load(blosum_path)['original_score'], \
                torch.load(blosum_path)['Qtb_temperature'], torch.load(blosum_path)['Qt_temperature']
        except FileNotFoundError:
            blosum_path = '../' + blosum_path
            self.original_score, self.temperature_list, self.Qt_temperature = torch.load(blosum_path)['original_score'], \
                torch.load(blosum_path)['Qtb_temperature'], torch.load(blosum_path)['Qt_temperature']
        self.X_classes = x_classes
        self.timestep = timestep
        temperature_list = self.temperature_list.unsqueeze(dim=0)
        temperature_list = temperature_list.unsqueeze(dim=0)
        Qt_temperature = self.Qt_temperature.unsqueeze(dim=0)
        Qt_temperature = Qt_temperature.unsqueeze(dim=0)
        if temperature_list.shape[0] != self.timestep:
            output_tensor = F.interpolate(temperature_list, size=timestep + 1, mode='linear', align_corners=True)
            self.temperature_list = output_tensor.squeeze()
            output_tensor = F.interpolate(Qt_temperature, size=timestep + 1, mode='linear', align_corners=True)
            self.Qt_temperature = output_tensor.squeeze()
        else:
            self.temperature_list = self.temperature_list
            self.Qt_temperature = self.Qt_temperature

    def get_Qt_bar(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.temperature_list = self.temperature_list.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.temperature_list[t_int.long()]
        q_x = self.original_score.unsqueeze(0) / temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x, dim=2)
        q_x[q_x < 1e-6] = 1e-6
        return q_x

    def get_Qt(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.Qt_temperature = self.Qt_temperature.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.Qt_temperature[t_int.long()]
        q_x = self.original_score.unsqueeze(0) / temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x, dim=2)
        return q_x
