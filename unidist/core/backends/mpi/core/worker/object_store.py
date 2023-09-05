# Copyright (C) 2021-2023 Modin authors
#
# SPDX-License-Identifier: Apache-2.0

import weakref

try:
    import mpi4py
except ImportError:
    raise ImportError(
        "Missing dependency 'mpi4py'. Use pip or conda to install it."
    ) from None


# TODO: Find a way to move this after all imports
mpi4py.rc(recv_mprobe=False, initialize=False)
from mpi4py import MPI  # noqa: E402

from unidist.core.backends.mpi.core.serialization import ComplexDataSerializer
import unidist.core.backends.mpi.core.common as common
import unidist.core.backends.mpi.core.communication as communication


mpi_state = communication.MPIState.get_instance()
# Logger configuration
# When building documentation we do not have MPI initialized so
# we use the condition to set "worker_0.log" in order to build it succesfully.
logger_name = "worker_{}".format(mpi_state.rank if mpi_state is not None else 0)
log_file = "{}.log".format(logger_name)
logger = common.get_logger(logger_name, log_file)


class ObjectStore:
    """
    Class that stores local objects and provides access to them.

    Notes
    -----
    For now, the storage is local to the current worker process only.
    """

    __instance = None

    def __init__(self):
        # Add local data {DataId : Data}
        self._data_map = weakref.WeakKeyDictionary()
        # "strong" references to data IDs {DataId : DataId}
        # we are using dict here to improve performance when getting an element from it,
        # whereas other containers would require O(n) complexity
        self._data_id_map = {}
        # Data owner {DataId : Rank}
        self._data_owner_map = weakref.WeakKeyDictionary()
        # Data serialized cache
        self._serialization_cache = {}

    @classmethod
    def get_instance(cls):
        """
        Get instance of ``ObjectStore``.

        Returns
        -------
        ObjectStore
        """
        if cls.__instance is None:
            cls.__instance = ObjectStore()
        return cls.__instance

    def put(self, data_id, data):
        """
        Put `data` to internal dictionary.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.
        data : object
            Data to be put.
        """
        self._data_map[data_id] = data

    def put_data_owner(self, data_id, rank):
        """
        Put data location (owner rank) to internal dictionary.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.
        rank : int
            Rank number where the data resides.
        """
        self._data_owner_map[data_id] = rank

    def get(self, data_id, force=False):
        """
        Get the data from a local dictionary.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.

        Returns
        -------
        object
            Return local data associated with `data_id`.
        """
        data = self._data_map[data_id]
        if isinstance(data, common.PendingRequest):
            if force:
                MPI.Request.Waitall(data.requests)
                is_ready = True
            else:
                is_ready = MPI.Request.Testall(data.requests)
            if is_ready:
                deserializer = ComplexDataSerializer(
                    data.raw_buffers, data.buffer_count
                )
                data = deserializer.deserialize(data.msgpack_buffer)["data"]
                self._data_map[data_id] = data
                from unidist.core.backends.mpi.core.worker.task_store import TaskStore
                task_store = TaskStore.get_instance()
                task_store.check_pending_tasks()
                task_store.check_pending_actor_tasks()
                return data
            else:
                return data
        return data

    def get_data_owner(self, data_id):
        """
        Get the data owner rank.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.

        Returns
        -------
        int
            Rank number where the data resides.
        """
        return self._data_owner_map[data_id]

    def contains(self, data_id):
        """
        Check if the data associated with `data_id` exists in a local dictionary.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.

        Returns
        -------
        bool
            Return the status if an object exist in local dictionary.
        """
        return data_id in self._data_map

    def contains_data_owner(self, data_id):
        """
        Check if the data location info associated with `data_id` exists in a local dictionary.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.

        Returns
        -------
        bool
            Return the ``True`` status if an object location is known.
        """
        return data_id in self._data_owner_map

    def get_unique_data_id(self, data_id):
        """
        Get the "strong" reference to the data ID if it is already stored locally.

        If the passed data ID is not stored locally yet, save and return it.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.

        Returns
        -------
        unidist.core.backends.common.data_id.DataID
            The unique ID to data.

        Notes
        -----
        We need to use a unique data ID reference for the garbage colleactor to work correctly.
        """
        if data_id not in self._data_id_map:
            self._data_id_map[data_id] = data_id
        return self._data_id_map[data_id]

    def clear(self, cleanup_list):
        """
        Clear "strong" references to data IDs from `cleanup_list`.

        Parameters
        ----------
        cleanup_list : list
            List of data IDs.

        Notes
        -----
        The actual data will be collected later when there is no weak or
        strong reference to data in the current worker.
        """
        # for data_id in cleanup_list:
        #     data = self.get(data_id)
        #     if not isinstance(data, common.PendingRequest):
        #         self._data_id_map.pop(data_id, None)

    def cache_serialized_data(self, data_id, data):
        """
        Save serialized object for this `data_id` and rank.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.
        data : object
            Serialized data to cache.
        """
        self._serialization_cache[data_id] = data

    def is_already_serialized(self, data_id):
        """
        Check if the data on this `data_id` is already serialized.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.

        Returns
        -------
        bool
            ``True`` if the data is already serialized.
        """
        return data_id in self._serialization_cache

    def get_serialized_data(self, data_id):
        """
        Get serialized data on this `data_id`.

        Parameters
        ----------
        data_id : unidist.core.backends.common.data_id.DataID
            An ID to data.

        Returns
        -------
        object
            Cached serialized data associated with `data_id`.
        """
        return self._serialization_cache[data_id]
