"""Classification metrics implemented from the definitions.

scikit-learn is available, but the metrics here are written out for two
reasons: the thresholds and tie handling are part of the project's design
decisions, and a reviewer can check the arithmetic.  Every function validates
its inputs, because silent metric bugs (single-class AUC, mismatched lengths)
are the kind that make a paper wrong.
"""
