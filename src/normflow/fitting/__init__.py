"""
Integration with zfit for fitting.
"""

from .zfit_interface import ZfitFlowInterface, ZfitLossWrapper
from .nll_loss import NLLLoss, ZfitNLLLoss, CustomNLLLoss

from . import zfit_interface, nll_loss