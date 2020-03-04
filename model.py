""" Models used in the experiments """
import math

import torch
import torch.nn as nn


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class ControllerMonolithic(torch.nn.Module):
    """Class that represents the monolithic network"""

    def __init__(self, D_in, H, D_out, min_std=1e-6, init_std=1.0, det=False):
        super(ControllerMonolithic, self).__init__()
        self.din = D_in
        self.dout = D_out
        self.controller = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, D_out),
        )
        self.sigma = nn.Parameter(torch.Tensor(D_out))
        self.sigma.data.fill_(math.log(init_std))

        self.min_log_std = math.log(min_std)
        self.saved_log_probs = []
        self.rewards = []
        self.deterministic = det

    def forward(self, x):
        means = self.controller(x)
        scales = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))
        dist = torch.distributions.Normal(means, scales)
        action = dist.mean if self.deterministic else dist.sample()
        log_prob = 0 if self.deterministic else dist.log_prob(action)
        return action, log_prob, None


class ControllerInstinct(torch.nn.Module):
    """Single element of the modular network"""

    def __init__(self, D_in, H, D_out):
        super(ControllerInstinct, self).__init__()
        self.din = D_in
        self.dout = D_out
        self.controller = nn.Sequential(
            nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU()
        )
        self.last_layer = nn.Sequential(nn.Linear(H, D_out), nn.Tanh())
        self.strength_signal = nn.Sequential(nn.Linear(H, D_out), nn.Sigmoid())

    def forward(self, x):
        inter = self.controller(x)
        means = self.last_layer(inter)
        str_sig = self.strength_signal(inter)
        return means, str_sig


class Controller(torch.nn.Module):
    """Single element of the modular network"""

    def __init__(self, D_in, H, D_out, min_std=1e-6, init_std=1.0):
        super(Controller, self).__init__()
        self.din = D_in
        self.dout = D_out
        self.controller = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, D_out),
            nn.Tanh(),
        )
        self.sigma = nn.Parameter(torch.Tensor(D_out))
        self.sigma.data.fill_(math.log(init_std))
        self.min_log_std = math.log(min_std)

        self.deterministic = False

    def forward(self, x):
        means = self.controller(x)
        scales = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))
        dist = torch.distributions.Normal(means, scales)
        action = dist.mean if self.deterministic else dist.sample()
        log_prob = 0 if self.deterministic else dist.log_prob(action)
        return action, log_prob


class ControllerInstinctSigma(torch.nn.Module):
    """Single element of the modular network"""

    def __init__(self, D_in, H, D_out):
        super(ControllerInstinctSigma, self).__init__()
        D_in += 1
        self.din = D_in
        self.controller = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, D_out),
            nn.Tanh(),
        )

        self.sigma_controller = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, D_out),
            nn.Sigmoid(),
        )

        self.controller.deterministic = False
        self.apply(weight_init)

    def forward(self, x):
        means = self.controller(x)
        sigmas = self.sigma_controller(x)
        dist = torch.distributions.Normal(means, sigmas)
        action = dist.mean if self.controller.deterministic else dist.sample()
        log_prob = 0 if self.controller.deterministic else dist.log_prob(action)
        return action, log_prob, sigmas

    def get_combinator_params(self):
        # comb_params = []
        # dct = self.named_parameters()
        # for pkey, ptensor in dct:
        #    if "combinator" in pkey or pkey == "sigma":
        #        comb_params.append(ptensor)
        # return comb_params
        # return self.controller.parameters()
        return super(ControllerInstinctSigma, self).parameters()

    def get_evolvable_params(self):
        return super(ControllerInstinctSigma, self).parameters()

    def parameters(self):
        return super(ControllerInstinctSigma, self).parameters()


class ControllerCombinator(torch.nn.Module):
    """ The combinator that is modified during lifetime"""

    def __init__(self, D_in, H, D_out, min_std=1e-6, init_std=0.1, load_instinct=False):
        super(ControllerCombinator, self).__init__()

        # Initialize the modules
        self.controller = Controller(D_in + 1, H, D_out, init_std=init_std)
        self.instinct = ControllerInstinct(D_in + 1, H, D_out)
        if load_instinct:
            pretrained_inst = torch.load("instinct.pt")
            self.instinct.load_state_dict(pretrained_inst.state_dict())

        # Initialize the combinator dimensions and the combinator
        combinator_input_size = 2

        self.combinators = nn.ModuleList()
        for _ in range(D_out):
            self.combinators.append(nn.Linear(combinator_input_size, 1))

        # self.combinator = nn.Sequential(nn.Linear(combinator_input_size, D_out))

        self.sigma = nn.Parameter(torch.Tensor(D_out))

        self.apply(weight_init)

        self.sigma.data.fill_(math.log(init_std))
        self.min_log_std = math.log(min_std)
        self.freeze_instinct = load_instinct

    def forward(self, x):

        # Pass the input to the submodules
        stoch_action, log_prob = self.controller(x)
        alt_action, control = self.instinct(x)

        controlled_stoch_action = stoch_action * control

        # Separate the output dimensions
        output_dims = [
            torch.tensor(ins) for ins in zip(controlled_stoch_action, alt_action)
        ]
        # comb_x = torch.cat((controlled_stoch_action, alt_action))

        # Combine actions into the final action
        output_list = [
            combinator(comb_x)
            for combinator, comb_x in zip(self.combinators, output_dims)
        ]
        final_action = torch.cat(output_list)

        return final_action, log_prob + torch.log(control), control

    def get_combinator_params(self):
        # comb_params = []
        # dct = self.named_parameters()
        # for pkey, ptensor in dct:
        #    if "combinator" in pkey or pkey == "sigma":
        #        comb_params.append(ptensor)
        # return comb_params
        return self.controller.parameters()

    def get_evolvable_params(self):
        comb_params = []
        dct = self.named_parameters()
        for pkey, ptensor in dct:
            if not ("instinct" in pkey and self.freeze_instinct):
                comb_params.append(ptensor)
        return comb_params

    def parameters(self):
        return super.parameters()

class ControllerNonParametricCombinator(torch.nn.Module):
    """ The combinator that is modified during lifetime"""

    def __init__(self, D_in, H, D_out, min_std=1e-6, init_std=0.1, load_instinct=False):
        super(ControllerNonParametricCombinator, self).__init__()

        # Initialize the modules
        self.controller = Controller(D_in, H, D_out, init_std=init_std)
        self.instinct = ControllerInstinct(D_in, H, D_out)
        if load_instinct:
            loaded_instinct = torch.load("instinct.pt")
            self.instinct.load_state_dict(self.instinct.state_dict())

        # Initialize the combinator dimensions and the combinator
        # combinator_input_size = 2

        # self.combinators = nn.ModuleList()
        # for _ in range(D_out):
        #     self.combinators.append(nn.Linear(combinator_input_size, 1))

        # self.combinator = nn.Sequential(nn.Linear(combinator_input_size, D_out))

        self.apply(weight_init)
        self.freeze_instinct = load_instinct

    def forward(self, x):

        # Pass the input to the submodules
        stoch_action, log_prob = self.controller(x)
        instinct_action, control = self.instinct(x)

        controlled_stoch_action = stoch_action * control
        controlled_instinct_action = instinct_action * (1 - control)

        final_action = controlled_stoch_action + controlled_instinct_action

        return final_action, log_prob + torch.log(control), control

    def get_combinator_params(self):
        # comb_params = []
        # dct = self.named_parameters()
        # for pkey, ptensor in dct:
        #    if "combinator" in pkey or pkey == "sigma":
        #        comb_params.append(ptensor)
        # return comb_params
        return self.controller.parameters()

    def get_evolvable_params(self):
        comb_params = []
        dct = self.named_parameters()
        for pkey, ptensor in dct:
            if not ("instinct" in pkey and self.freeze_instinct):
                comb_params.append(ptensor)
        return comb_params

    def parameters(self):
        return super(ControllerNonParametricCombinator, self).parameters()


def init_model(din, dout, args):
    """ Method that gives instantiates the model depending on the program arguments """

    if args.parametric_combinator:
        model = ControllerCombinator(
            D_in=din,
            H=100,
            D_out=dout,
            init_std=args.init_sigma,
            load_instinct=args.load_instinct,
        )
    elif args.instinct_sigma:
        model = ControllerInstinctSigma(D_in=din, H=100, D_out=dout)
    else:
        model = ControllerNonParametricCombinator(
            D_in=din,
            H=100,
            D_out=dout,
            init_std=args.init_sigma,
            load_instinct=args.load_instinct,
        )
    return model
