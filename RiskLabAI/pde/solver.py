"""
Implements the main FBSDE (Forward-Backward Stochastic Differential
Equation) solver classes.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from typing import List, Tuple, Any

from RiskLabAI.pde.model import *
from RiskLabAI.pde.equation import Equation

def initialize_weights(m: nn.Module) -> None:
    """
    Initializes the weights of a Linear layer.

    Parameters
    ----------
    m : nn.Module
        The module to initialize.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0, 0.1)
        m.bias.data.fill_(0.00)


class FBSDESolver:
    """
    Solver for FBSDEs using various deep learning methods.
    """
    def __init__(
        self,
        pde: Equation,
        layer_sizes: List[int],
        learning_rate: float,
        solving_method: str,
        device: torch.device,
    ):
        """
        Initialize the FBSDE Solver.

        Parameters
        ----------
        pde : Equation
            The PDE to solve (from equation.py).
        layer_sizes : List[int]
            List of hidden layer sizes for the network.
        learning_rate : float
            Learning rate for the optimizer.
        solving_method : str
            Method to use: 'Monte-Carlo', 'Deep-Time-SetTransformer',
            'DTNN', or 'DeepBSDE'.
        device : torch.device
            The device ('cpu' or 'cuda') to run on.
        """
        self.pde = pde
        self.layer_size = layer_sizes
        self.learning_rate = learning_rate
        self.method = solving_method
        self.device = device

        # 1. Initialize the correct network model
        if solving_method == 'Monte-Carlo':
            self.solver = TimeDependentNetworkMonteCarlo(
                pde.dim, self.layer_size, pde.dim, pde.sigma
            ).to(device)
        elif solving_method == 'Deep-Time-SetTransformer':
            self.solver = DeepTimeSetTransformer(1).to(device) # Assumes dim=1
        elif solving_method == 'DTNN':
            self.solver = TimeDependentNetwork(
                pde.dim, self.layer_size, pde.dim
            ).to(device)
        elif solving_method == 'DeepBSDE':
            # Create a separate network for each time step
            self.solver = nn.ModuleList(
                [DeepBSDE(layer_sizes).to(device) for _ in range(pde.num_time_interval)]
            )
        else:
            raise ValueError(f"Unknown solving_method: {solving_method}")

        # 2. Initialize the optimizer
        if solving_method != 'DeepBSDE':
            self.optimizer = torch.optim.Adam(
                self.solver.parameters(), lr=self.learning_rate, betas=(0.9, 0.99)
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.solver.parameters(), lr=self.learning_rate, betas=(0.9, 0.99)
            )

    def compute_loss(
        self, y_path: Tensor, dw_path: Tensor, t: Tensor, init_y: Tensor, init_z: Tensor
    ) -> Tuple[Tensor, ...]:
        """
        Compute the loss for a single batch.
        """
        batch_size = y_path.size()[0]
        y_terminal = init_y.expand(batch_size, 1)
        
        coef = torch.ones((batch_size, 1), device=self.device)
        dw_coef = torch.zeros((batch_size, 1), device=self.device)

        for z in range(self.pde.num_time_interval):
            S0 = y_path[:, :, z]
            t0 = t * self.pde.delta_t * z

            if self.method == 'DeepBSDE':
                if z > 0:
                    out_z = self.solver[z](S0)
                else:
                    # Use initial gradient Z_0
                    out_z = init_z.expand(batch_size, self.pde.dim)
            else:
                # Other models are time-dependent
                out_z = self.solver(t0, S0) # This assumes other models return Z
            
            samp_dw = dw_path[:, :, z]
            
            # Get driver components
            interest_rate = self.pde.r_u(t0, S0, y_terminal, out_z)
            hz = self.pde.h_z(t0, S0, y_terminal, out_z)
            
            # Update path
            y_terminal = (
                y_terminal * (1 + interest_rate * self.pde.delta_t)
                + hz * self.pde.delta_t
                + torch.sum(out_z * samp_dw, dim=1, keepdim=True)
            )
            
            # Update coefficients for loss calculation
            if z > 0:
                dw_coef = dw_coef * (1 + interest_rate * self.pde.delta_t) + torch.sum(
                    out_z * samp_dw, dim=1, keepdim=True
                ) + hz * self.pde.delta_t
            else:
                dw_coef = torch.sum(out_z * samp_dw, dim=1, keepdim=True) + hz * self.pde.delta_t
            
            coef = coef * (1 + interest_rate * self.pde.delta_t)

        # 4. Calculate terminal payoff and loss
        payoff = self.pde.terminal(
            t * self.pde.total_time, y_path[:, :, -1]
        )
        
        loss = torch.mean(torch.square(payoff - y_terminal))
        
        return loss, coef, dw_coef, payoff

    def solve(
        self, num_iterations: int, batch_size: int, init_y: float
    ) -> Tuple[List[float], List[float]]:
        """
        Solves the PDE.

        Parameters
        ----------
        num_iterations : int
            Number of training iterations.
        batch_size : int
            Batch size for training.
        init_y : float
            The initial guess for Y_0 (the price at t=0).

        Returns
        -------
        Tuple[List[float], List[float]]
            - losses: List of losses at each iteration.
            - inits: List of Y_0 values at each iteration.
        """
        losses = []
        inits = []
        
        # Y_0 and Z_0 are trainable parameters for DeepBSDE
        y0 = torch.tensor([init_y], device=self.device).requires_grad_(True)
        z0 = torch.zeros(1, self.pde.dim, device=self.device).requires_grad_(True)
        
        if self.method == 'DeepBSDE':
            init_opt = torch.optim.Adam([y0], lr=0.1, betas=(0.9, 0.99))
            init_grad_opt = torch.optim.Adam([z0], lr=0.001, betas=(0.9, 0.99))

        # Validation data
        dw_val, y_val = self.pde.sample(128)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        dw_val = torch.tensor(dw_val, dtype=torch.float32, device=self.device)
        t_val = torch.ones((128, 1), device=self.device)

        # Set model to train mode
        if self.method != 'DeepBSDE':
            self.solver.train()
        else:
            for model in self.solver:
                model.train()

        for j in range(num_iterations):
            print(f"Iteration {j + 1}/{num_iterations}")

            dw_train, y_train = self.pde.sample(batch_size)
            y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
            dw_train = torch.tensor(dw_train, dtype=torch.float32, device=self.device)
            t_train = torch.ones((batch_size, 1), device=self.device)

            self.optimizer.zero_grad()
            if self.method == 'DeepBSDE':
                init_opt.zero_grad()
                init_grad_opt.zero_grad()

            loss, _, _, _ = self.compute_loss(
                y_train, dw_train, t_train, y0, z0
            )
            
            loss.backward()
            
            self.optimizer.step()
            if self.method == 'DeepBSDE':
                init_opt.step()
                init_grad_opt.step()

            # Validation and logging
            with torch.no_grad():
                val_loss, coef, dw_coef, payoff = self.compute_loss(
                    y_val, dw_val, t_val, y0, z0
                )
                
                if self.method != 'DeepBSDE':
                    # For non-DeepBSDE, Y_0 is not a param but computed
                    # from the loss function's components.
                    y0_new = torch.mean((payoff - dw_coef) / coef)
                    inits.append(y0_new.item())
                    print(f"Loss: {val_loss.item():.4f}, Y_0: {y0_new.item():.4f}")
                else:
                    inits.append(y0.item())
                    print(f"Loss: {val_loss.item():.4f}, Y_0: {y0.item():.4f}")
                    
                losses.append(val_loss.item())

        return losses, inits


class FBSNNolver:
    """
    Solver for FBSNN (Forward-Backward Stochastic Neural Network).
    
    This method solves the PDE by learning the solution Y_t directly
    at each time step and minimizing the difference between the
    predicted Y_{t+1} and the one-step-ahead approximation.
    """
    def __init__(
        self,
        pde: Equation,
        layer_sizes: List[int],
        learning_rate: float,
        device: torch.device
    ):
        """
        Initialize the FBSNN Solver.

        Parameters
        ----------
        pde : Equation
            The PDE to solve.
        layer_sizes : List[int]
            Layer sizes for the FBSNNNetwork.
        learning_rate : float
            Learning rate for the optimizer.
        device : torch.device
            The device ('cpu' or 'cuda') to run on.
        """
        self.pde = pde
        self.layer_size = layer_sizes
        self.learning_rate = learning_rate
        self.device = device

        self.solver = FBSNNNetwork(self.layer_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.solver.parameters(), lr=self.learning_rate, betas=(0.99, 0.999)
        )

    def compute_loss(
        self, y_path: Tensor, dw_path: Tensor, t: Tensor, init_y: Tensor
    ) -> Tensor:
        """
        Compute the loss for the FBSNN method.
        """
        batch_size = y_path.size()[0]
        y_terminal = init_y.expand(batch_size, 1) # Not used, but kept for signature
        loss = torch.tensor(0.0, device=self.device)

        for z in range(self.pde.num_time_interval):
            S0 = y_path[:, :, z].requires_grad_(True)
            S1 = y_path[:, :, z + 1]

            t0 = t * self.pde.delta_t * z
            t1 = t * self.pde.delta_t * (z + 1)

            # 1. Get network output Y_t and Y_{t+1}
            y_0 = self.solver(torch.cat((t0, S0), dim=1))
            y_1 = self.solver(torch.cat((t1, S1), dim=1))

            # 2. Calculate the gradient Z_t = dY_t/dS_t * sigma(S_t)
            grad_y0 = autograd.grad(
                outputs=torch.sum(y_0), 
                inputs=S0, 
                create_graph=True
            )[0]
            Z = grad_y0 * self.pde.sigma_matrix(S0)

            samp_dw = dw_path[:, :, z]

            # 3. Get driver components
            interest_rate = self.pde.r_u(t0, S0, y_0, Z)
            hz = self.pde.h_z(t0, S0, y_0, Z)
            
            # 4. Calculate the one-step approximation Y_hat_{t+1}
            y_1_hat = (
                y_0 * (1 + interest_rate * self.pde.delta_t)
                + hz * self.pde.delta_t
                + torch.sum(Z * samp_dw, dim=1, keepdim=True)
            )
            
            # 5. Add to total loss
            loss += torch.mean(torch.square(y_1_hat - y_1))

        # 6. Add terminal condition loss
        S_T = y_path[:, :, -1]
        t_T = t * self.pde.total_time
        
        payoff = self.pde.terminal(t_T, S_T)
        y_T = self.solver(torch.cat((t_T, S_T), dim=1))
        
        loss += torch.mean(torch.square(payoff - y_T))
        
        return loss

    def solve(
        self,
        num_iterations: int,
        batch_size: int,
        init_y: float, # Note: init_y is not used by FBSNN, but kept for API
    ) -> Tuple[List[float], List[float]]:
        """
        Solves the PDE using the FBSNN method.

        Parameters
        ----------
        num_iterations : int
            Number of training iterations.
        batch_size : int
            Batch size for training.
        init_y : float
            Initial value (not used by this solver).

        Returns
        -------
        Tuple[List[float], List[float]]
            - losses: List of losses at each iteration.
            - inits: List of Y_0 values at each iteration.
        """
        losses = []
        inits = []
        
        self.solver.train()
        
        # Validation data
        dw_val, y_val = self.pde.sample(128)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        dw_val = torch.tensor(dw_val, dtype=torch.float32, device=self.device)
        t_val = torch.ones((128, 1), device=self.device)

        for j in range(num_iterations):
            print(f"Iteration {j + 1}/{num_iterations}")

            dw_train, y_train = self.pde.sample(batch_size)
            y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
            dw_train = torch.tensor(dw_train, dtype=torch.float32, device=self.device)
            t_train = torch.ones((batch_size, 1), device=self.device)

            self.optimizer.zero_grad()
            
            # init_y is not used, pass a dummy tensor
            dummy_init = torch.tensor(0.0, device=self.device)
            
            loss = self.compute_loss(y_train, dw_train, t_train, dummy_init)
            loss.backward()
            self.optimizer.step()

            # Validation and logging
            with torch.no_grad():
                val_loss = self.compute_loss(y_val, dw_val, t_val, dummy_init)
                losses.append(val_loss.item())

                # Get Y_0 from the trained network
                S0_val = y_val[:, :, 0]
                t0_val = t_val * self.pde.delta_t * 0
                y0_val = self.solver(torch.cat((t0_val, S0_val), dim=1))
                
                y0_mean = torch.mean(y0_val).item()
                inits.append(y0_mean)
                
                print(f"Loss: {val_loss.item():.4f}, Y_0: {y0_mean:.4f}")

        return losses, inits