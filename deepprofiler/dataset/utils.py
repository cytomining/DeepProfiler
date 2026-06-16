"""Shared utilities: timing, progress reporting, parallel processing, logging."""

import logging
import multiprocessing
import os
import sys
import time

PI = 3.1415926539


def print_progress(iteration, total, prefix="Progress", suffix="Complete", decimals=1, barLength=50):
    """Print an in-place ASCII progress bar to stdout.

    Args:
        iteration: Current iteration (int, 0-indexed).
        total: Total number of iterations (int).
        prefix: Label printed before the bar.
        suffix: Label printed after the percentage.
        decimals: Decimal places in the percentage display.
        barLength: Character width of the bar.
    """
    if all(t >= 0 for t in [iteration, total, barLength]) and iteration <= total:
        formatStr = "{0:." + str(decimals) + "f}"
        percents = formatStr.format(100 * (iteration / float(total)))
        filledLength = int(round(barLength * iteration / float(total)))
        bar = "#" * filledLength + "-" * (barLength - filledLength)
        sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),
        sys.stdout.flush()
        if iteration == total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    elif sum([iteration < 0, total < 0, barLength < 0]) > 1:
        sys.stdout.write("\rError: print_progress() function received multiple negative values.")
        sys.stdout.flush()
    elif iteration < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'iteration' value.")
        sys.stdout.flush()
    elif total < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'total' value.")
        sys.stdout.flush()
    elif barLength < 0:
        sys.stdout.write("\rError: print_progress() function received a negative 'barLength' value.")
        sys.stdout.flush()
    elif iteration > total:
        sys.stdout.write("\rError: print_progress() function received an 'iteration' value greater than the 'total' value.")
        sys.stdout.flush()


def check_path(filename):
    """Ensure the parent directory of ``filename`` exists, creating it if needed."""
    path = "/".join(filename.split("/")[0:-1])
    os.makedirs(path, exist_ok=True)


def tic():
    """Return the current wall-clock time in seconds (for use with :func:`toc`)."""
    return time.time()


def toc(msg, beginning):
    """Print elapsed time since ``beginning`` and return the current time.

    Args:
        msg: Label printed before the elapsed time.
        beginning: Start time returned by :func:`tic`.

    Returns:
        Current wall-clock time in seconds.
    """
    end = time.time()
    elapsed = end - beginning
    print(msg, ": {:.2f} secs".format(elapsed))
    return end


class Parallel():
    """Thin wrapper around :class:`multiprocessing.Pool` for map-style parallelism.

    Each call to :meth:`compute` invokes ``operation([item, fixed_args])`` for
    every item in ``data`` across the worker pool.  The fixed args pattern
    lets callers pass configuration or other shared state without using global
    variables.

    Args:
        fixed_args: A single object passed as the second element of every
            worker invocation — typically the experiment config dict.
        numProcs: Number of worker processes.  Defaults to CPU count; clamped
            to ``[1, cpu_count]``.
    """

    def __init__(self, fixed_args, numProcs=None):
        self.fixed_args = fixed_args
        cpus = multiprocessing.cpu_count()
        if numProcs is None or numProcs > cpus or numProcs < 1:
            numProcs = cpus
        self.pool = multiprocessing.Pool(numProcs)

    def compute(self, operation, data):
        """Run ``operation`` on every element of ``data`` in parallel.

        Args:
            operation: Callable that accepts ``[item, fixed_args]``.
            data: Iterable of items to process.

        Returns:
            List of results in the same order as ``data``.
        """
        iterable = [[d, self.fixed_args] for d in data]
        return self.pool.map(operation, iterable)

    def close(self):
        """Shut down the worker pool, waiting for all jobs to finish."""
        self.pool.close()
        self.pool.join()


class Logger():
    """Thin wrapper around the Python standard ``logging`` module.

    Configures a single ``INFO``-level handler writing to stdout with a
    timestamp prefix.  Instantiated once at module level as :data:`logger`.
    """

    def __init__(self):
        self.root = logging.getLogger()
        self.root.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        self.root.addHandler(ch)

    def log(self, level, msg):
        """Emit a log record at an arbitrary level."""
        self.root.log(level, msg)

    def info(self, msg):
        """Emit an INFO-level log record."""
        self.root.info(msg)


logger = Logger()
