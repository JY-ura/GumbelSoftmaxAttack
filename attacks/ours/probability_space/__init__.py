"""
This module contains the implementation of the probability space attack.
By optimizing the parameter of the probability, we turn the discrete categorical attack into continuous.
"""
from .event_generator import GumbelSoftmaxCustom, GumbelSoftmaxTorch, TempretureSoftmax
from .frame_generator import FrameGenerator
from .probability_attack import ProbabilityAttacker
