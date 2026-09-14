"""ranch -- a small explicit API over the granch_fast exact-inference core
(ENGINEERING_PLAN.md §2, Phase B). The math lives unchanged in granch_fast; this package
adds the World/Learner split, a registry of named decision variables, typed paradigm
results and linking protocols. Every runner here is proven identical to the engine in
granch_fast/tests/test_api_identity.py.
"""
from .config import Prior, LearnerNoise, Quadrature, Model
from .decision import EIG, EIGWithin, KL, Surprisal, RealizedGain, ALL_VARIABLES, DecisionVariable
from .world import World
from .learner import Learner
from .paradigms import forced_exposure_then_test, self_paced, LucePolicy, Result
from .canonical import CANONICAL, PUBLISHED_CORRECTED, PUBLISHED_SPEC, CONFIGURATIONS, NamedConfiguration
from .selection import Selection, UnquotableSelection, select_infant, select_adult, reevaluate_infant, reevaluate_adult
from . import linking, data, settings, pipeline, selection

__all__ = ["Prior", "LearnerNoise", "Quadrature", "Model", "EIG", "EIGWithin", "KL", "Surprisal",
           "RealizedGain", "ALL_VARIABLES", "DecisionVariable", "World", "Learner",
           "forced_exposure_then_test", "self_paced", "LucePolicy", "Result", "linking", "data", "settings", "pipeline",
           "selection", "Selection", "UnquotableSelection", "select_infant", "select_adult", "reevaluate_infant",
           "reevaluate_adult", "CANONICAL", "PUBLISHED_CORRECTED", "PUBLISHED_SPEC", "CONFIGURATIONS", "NamedConfiguration"]
