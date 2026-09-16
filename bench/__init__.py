"""The benchmark suite.

A package rather than a bag of scripts for one reason: ``bench/workloads/`` is imported by the
three mandatory tests, which spec task 5.1 requires to run "for every workload in
``bench/workloads/``". The benchmarks and the tests therefore share one definition of what the
workloads are, instead of drifting into two.
"""
