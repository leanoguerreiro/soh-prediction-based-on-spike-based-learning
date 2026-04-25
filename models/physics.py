import torch
import torch.nn as nn

from .transformers import PyTorch_iTransformer


# ---------------------------------------------------------------------------
# 1. PyTorch_Phys_iTransformer (Lambda Fixo)
# ---------------------------------------------------------------------------
class PyTorch_Phys_iTransformer(nn.Module):
    def __init__(self, time_steps, n_features, d_model=64):
        super().__init__()
        self.base_model = PyTorch_iTransformer(time_steps, n_features, d_model)
        self.phys_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        soh_pred = self.base_model(x)

        # Ramo da Física: Soma cumulativa da Corrente Absoluta (Corrente = index 1)
        curr = x[:, :, 1]
        cum_abs_curr = torch.sum(torch.abs(curr), dim=1, keepdim=True)  # [Batch, 1]
        phys_pred = self.phys_head(cum_abs_curr)

        return soh_pred, phys_pred


# ---------------------------------------------------------------------------
# 2. PyTorch_Phys_iTransformer_Curriculum (Lambda Adaptativo)
# ---------------------------------------------------------------------------
class PyTorch_Phys_iTransformer_Curriculum(nn.Module):
    """
    Phys-iTransformer com lambda adaptativo.
    O peso do termo físico sobe linearmente de `lambda_start` para `lambda_max`
    ao longo de `warmup_epochs`. A época atual é injetada via set_epoch().
    """

    def __init__(self, time_steps, n_features, d_model=64, lambda_start=0.0, lambda_max=2.0, warmup_epochs=50):
        super().__init__()
        self.base_model = PyTorch_iTransformer(time_steps, n_features, d_model)
        self.phys_head = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.lambda_start = lambda_start
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        """Chamado pelo loop de treino a cada época."""
        self._current_epoch = epoch

    def get_lambda(self) -> float:
        """Retorna o lambda atual interpolado linearmente."""
        progress = min(self._current_epoch / max(self.warmup_epochs, 1), 1.0)
        return self.lambda_start + progress * (self.lambda_max - self.lambda_start)

    def forward(self, x):
        soh_pred = self.base_model(x)

        # Índice 1 corresponde a 'Current_measured'
        cum_abs_curr = torch.sum(torch.abs(x[:, :, 1]), dim=1, keepdim=True)
        phys_pred = self.phys_head(cum_abs_curr)

        return soh_pred, phys_pred
