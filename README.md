# Pulse Design

Simple pulse design algorithm using [TokaMaker](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) and [Torax](https://github.com/google-deepmind/torax).

To install:
`python -m venv env`

`source env/bin/activate`

`pip install torax`

Then clone and install [OpenFUSIONTookit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) using instructions in the OFT documentation.

To run e.g. example 3:
`mkdir tmp`

`python example3.py`

To track simulation progress (consumed flux):

`tail -f convergence_history.txt`

Note: simulation has oustanding issues with consumed flux calculation.

Once simulation is finished visualization scripts can be used, e.g.

`python vis/heatingplot.py`
