"""Visualization helpers and Manim scenes for the representation-learning lecture."""
from .dot_product             import show_dot_product,             DotProductScene
from .softmax_viz             import show_softmax,                 SoftmaxScene
from .triplet_viz             import show_triplet,                 TripletScene
from .representation_collapse import show_representational_collapse, RepresentationCollapseScene
from .triplet_concept         import show_triplet_concept,           TripletConceptScene

__all__ = [
    "show_dot_product",               "DotProductScene",
    "show_softmax",                   "SoftmaxScene",
    "show_triplet",                   "TripletScene",
    "show_representational_collapse", "RepresentationCollapseScene",
    "show_triplet_concept",           "TripletConceptScene",
]
