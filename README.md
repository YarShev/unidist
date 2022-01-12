<p align="center"><a href="https://unidist.readthedocs.io"><img width=77% alt="" src="https://github.com/modin-project/unidist/docs/img/unidist_logo.png?raw=true"></a></p>
<h2 align="center">Unified Distributed Execution</h2>

<p align="center">
<a href="https://github.com/modin-project/modin/actions"><img src="https://github.com/modin-project/modin/workflows/master/badge.svg" align="center"></a>
<a href="https://unidist.readthedocs.io/en/latest/?badge=latest"><img alt="" src="https://readthedocs.org/projects/unidist/badge/?version=latest" align="center"></a>
</p>

### What is unidist?

``unidist`` is the framework that is intended to provide the unified API for distributed execution by supporting various execution backends. At the moment the following backends are supported under the hood:

* [Ray](https://docs.ray.io/en/master/index.html)
* [MPI](https://www.mpi-forum.org/)
* [Dask Distributed](https://distributed.dask.org/en/latest/)
* [Python Multiprocessing](https://docs.python.org/3/library/multiprocessing.html)

``unidist`` is designed to work in a [task-based parallel](https://en.wikipedia.org/wiki/Task_parallelism) model.

#### Choosing an execution backend

```bash
export UNIDIST_BACKEND=ray  # unidist will use Ray
```

#### Usage

```python
import ray
ray.init(plasma_directory="/path/to/custom/dir", object_store_memory=10**10)
# Modin will connect to the existing Ray environment
import modin.pandas as pd
```

### Full Documentation

Visit the complete documentation on readthedocs: https://unidist.readthedocs.io
