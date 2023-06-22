# Copyright (C) 2021-2023 Modin authors
#
# SPDX-License-Identifier: Apache-2.0

try:
    import mpi4py
except ImportError:
    raise ImportError(
        "Missing dependency 'mpi4py'. Use pip or conda to install it."
    ) from None

# TODO: Find a way to move this after all imports
mpi4py.rc(recv_mprobe=False, initialize=False)
from mpi4py import MPI  # noqa: E402

import unidist.core.backends.mpi.core.common as common

logger = common.get_logger("async_operations", "async_operations.log")


class AsyncOperations:
    """
    Class that stores MPI async communication handlers.

    Class holds a reference to sending data to prolong data lifetime during send operation.
    """

    __instance = None

    def __init__(self):
        # I-prefixed mpi call handlers
        self._send_async_handlers = []

    @classmethod
    def get_instance(cls):
        """
        Get instance of ``AsyncOperations``.

        Returns
        -------
        AsyncOperations
        """
        if cls.__instance is None:
            cls.__instance = AsyncOperations()
        return cls.__instance

    def extend(self, handlers_list):
        """
        Extend internal list with `handler_list`.

        Parameters
        ----------
        handler_list : list
            A list of pairs with handler and data reference.
        """
        self._send_async_handlers.append(handlers_list)

    def check(self):
        """Check all MPI async send requests readiness and remove a reference to sending data."""

        def is_ready(handler_list):
            is_ready = MPI.Request.Testall([h for h, _ in handler_list])
            if is_ready:
                logger.debug("CHECK ASYNC HANDLER {} - ready".format(is_ready))
            else:
                logger.debug("CHECK ASYNC HANDLER {} - not ready".format(is_ready))
            return is_ready

        # tup[0] - mpi async send handler object
        self._send_async_handlers[:] = [
            hl for hl in self._send_async_handlers if not is_ready(hl)
        ]

    def finish(self):
        """Cancel all MPI async send requests."""
        # We intentionaly iterate inversely so that an "intermediate" send
        # doesn't match the recv of an operation type. "intermediate" implies
        # any send in a complex communication other than the send of an operation type.
        for handler_list in self._send_async_handlers[::-1]:
            for handler, data in handler_list:
                # If data is None, that indicates we used a lower-case MPI-routine
                # to send a regular Python object. This is why we use
                # lower-case methods here to cancel async send requests.
                # Otherwise, if we send a buffer-like object, we use upper-case methods.
                # See more in https://mpi4py.readthedocs.io/en/latest/tutorial.html.
                if data is None:
                    handler.cancel()
                    handler.wait()
                else:
                    handler.Cancel()
                    handler.Wait()
        self._send_async_handlers.clear()
