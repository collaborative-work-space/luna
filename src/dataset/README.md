# Dataset loaders

`panopli.py` is the loader wired up in `dataset/__init__.py` (`dataset_class: panopli`) and is
what `config/panopli.yaml` uses by default. It expects per-frame `color/*.png` and `depth/*.png`.

`panopli-dark.py`, `panopli-geo.py`, and `panopli-noise.py` are loader variants used for the
degraded-Replica experiments in the paper (Sec. IV-A), where exported frames are `.jpg` and depth
is stored/read slightly differently. They are not dispatched automatically — to reproduce a
specific degradation run, swap the relevant file in as `panopli.py` (or edit `get_dataset` in
`__init__.py` to add a new `dataset_class` branch pointing at it) before launching training.
