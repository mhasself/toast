# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys
from collections.abc import MutableMapping

from .mpi import MPI

from .instrument import Telescope, Focalplane


class Observation(MutableMapping):
    """Class representing the data for one observation.

    Transitional API:  Currently this class is used to provide a future-compatible
    interface to the existing observation dictionary containing a TOD instance.  Until
    operators are ported to use this new API, an Observation class should be constructed
    by calling the "wrap_old" function on an existing "old" observation dictionary.

    Args:

    """

    def __init__(
        self,
        telescope,
        samples=None,  # The total number of samples.
        detector_ranks=1,  # The detector dimension of the process grid.
        detector_breaks=None,  # The forced breaks between sets of detectors.
        sample_sizes=None,  # The optional discrete chunking in time.
        sample_breaks=None,  # The forced breaks between the chunks.
        mpicomm=None,  # The optional MPI communicator.
        keynames=None,  # Optionally override the key names for standard keys.
        old_tod=None,
    ):
        self._telescope = telescope
        self._keynames = keynames
        if self._keynames is None:
            # Use the default names for timestream objects
            self._keynames = {
                "times": "times",
                "common": "common",
                "common_flags": "common_flags",
                "velocity": "velocity",
                "position": "position",
                "signal": "signal",
                "flags": "flags",
                "boresight": "boresight",
                "hwp_angle": "hwp_angle",
            }

        if old_tod is not None:
            # get all info from old TOD class
            self._old_tod = old_tod
            self._mpicomm = old_tod.mpicomm
            self._samples = old_tod.total_samples
            self._detector_ranks = old_tod._detranks
            self._sample_sizes = old_tod.total_chunks
            self._sample_ranks = old_tod._sampranks
            self._rank_det = old_tod._rank_det
            self._rank_samp = old_tod._rank_samp
            self._comm_row = old_tod._comm_row
            self._comm_col = old_tod._comm_col
            self._dist_dets = old_tod._dist_dets
            self._dist_samples = old_tod._dist_samples
            self._dist_sizes = old_tod._dist_sizes
        else:
            self._samples = samples
            self._detector_ranks = detector_ranks
            self._sample_sizes = sample_sizes
            self._mpicomm = mpicomm

            self._sample_ranks = 1
            self._rank_det = 0
            self._rank_samp = 0
            self._comm_row = None
            self._comm_col = None

            rank = 0

            if mpicomm is None:
                if detranks != 1:
                    raise RuntimeError("MPI is disabled, so detranks must equal 1")
            else:
                rank = mpicomm.rank
                if mpicomm.size % detranks != 0:
                    raise RuntimeError(
                        "The number of detranks ({}) does not divide evenly into the "
                        "communicator size ({})".format(detranks, mpicomm.size)
                    )
                self._sample_ranks = mpicomm.size // detranks
                self._rank_det = mpicomm.rank // self._sample_ranks
                self._rank_samp = mpicomm.rank % self._sample_ranks

                # Split the main communicator into process row and column
                # communicators, since this is useful for gathering data in some
                # operations.

                if self._sample_ranks == 1:
                    self._comm_row = MPI.COMM_SELF
                else:
                    self._comm_row = self._mpicomm.Split(
                        self._rank_det, self._rank_samp
                    )

                if self._detector_ranks == 1:
                    self._comm_col = MPI.COMM_SELF
                else:
                    self._comm_col = self._mpicomm.Split(
                        self._rank_samp, self._rank_det
                    )

            # if sizes is specified, it must be consistent with
            # the total number of samples.
            if self._sample_sizes is not None:
                test = np.sum(self._sample_sizes)
                if samples != test:
                    raise RuntimeError(
                        "Sum of sample_sizes ({}) does not equal total samples ({})".format(
                            test, samples
                        )
                    )

            (
                self._dist_dets,
                self._dist_samples,
                self._dist_sizes,
            ) = distribute_samples(
                self._mpicomm,
                self._telescope.focalplane.detectors,
                self._samples,
                detranks=self._detector_ranks,
                detbreaks=detector_breaks,
                sampsizes=self._sample_sizes,
                sampbreaks=sample_breaks,
            )

            if self._sample_sizes is None:
                # in this case, the chunks just come from the uniform distribution.
                self._sample_sizes = [
                    self._dist_samples[x][1] for x in range(self._sample_ranks)
                ]

            if rank == 0:
                # check that all processes have some data, otherwise print warning
                for d in range(self._detector_ranks):
                    if len(self._dist_dets[d]) == 0:
                        print(
                            "WARNING: detector rank {} has no detectors"
                            " assigned in observation.".format(d)
                        )
                for r in range(self._sample_ranks):
                    if self._dist_samples[r][1] <= 0:
                        print(
                            "WARNING: sample rank {} has no data assigned "
                            "in observation.".format(r)
                        )

        self._internal = dict()

    @property
    def telescope(self):
        return self._telescope

    @property
    def mpicomm(self):
        """
        (mpi4py.MPI.Comm): the group communicator assigned to this observation.
        """
        return self._mpicomm

    @property
    def grid_size(self):
        """
        (tuple): the dimensions of the process grid in (detector, sample)
            directions.
        """
        return (self._detranks, self._sampranks)

    @property
    def grid_ranks(self):
        """
        (tuple): the ranks of this process in the (detector, sample)
            directions.
        """
        return (self._rank_det, self._rank_samp)

    @property
    def grid_comm_row(self):
        """
        (mpi4py.MPI.Comm): a communicator across all detectors in the same
            row of the process grid (or None).
        """
        return self._comm_row

    @property
    def grid_comm_col(self):
        """
        (mpi4py.MPI.Comm): a communicator across all detectors in the same
            column of the process grid (or None).
        """
        return self._comm_col

    @property
    def samples(self):
        """(int): the total number of samples in this observation.
        """
        return self._samples

    @property
    def dist_samples(self):
        """
        (list): This is a list of 2-tuples, with one element per column
            of the process grid.  Each tuple is the same information
            returned by the "local_samples" member for the corresponding
            process grid column rank.
        """
        return self._dist_samples

    @property
    def local_samples(self):
        """
        (2-tuple): The first element of the tuple is the first global
            sample assigned to this process.  The second element of
            the tuple is the number of samples assigned to this
            process.
        """
        return self._dist_samples[self._rank_samp]

    @property
    def total_chunks(self):
        """
        (list): the full list of sample chunk sizes that were used in the
            data distribution.
        """
        return self._sizes

    @property
    def dist_chunks(self):
        """
        (list): this is a list of 2-tuples, one for each column of the process
        grid.  Each element of the list is the same as the information returned
        by the "local_chunks" member for a given process column.
        """
        return self._dist_sizes

    @property
    def local_chunks(self):
        """
        (2-tuple): the first element of the tuple is the index of the
        first chunk assigned to this process (i.e. the index in the list
        given by the "total_chunks" member).  The second element of the
        tuple is the number of chunks assigned to this process.
        """
        return self._dist_sizes[self._rank_samp]

    @property
    def detectors(self):
        """
        (list): Convenience wrapper for telescope.focalplane.detectors
        """
        return self._telescope.focalplane.detectors

    @property
    def local_detectors(self):
        """
        (list): The detectors assigned to this process.
        """
        return self._dist_dets[self._rank_det]

    @property
    def keynames(self):
        """
        (dict): The dictionary keys used for standard timestream products.
        """
        return self._keynames

    def times(self, name=None):
        if self._old_tod is None:
            raise NotImplementedError(
                "The Observation class currently only wraps a legacy TOD"
            )
        else:
            # We are wrapping.  First ensure data is cached
            _ = self._old_tod.local_times()
            # Now return a reference to the cached data
            cachename = self._keynames["times"]
            return self._old_tod.cache.reference(cachename)

    def velocity(self, name=None):
        if self._old_tod is None:
            raise NotImplementedError(
                "The Observation class currently only wraps a legacy TOD"
            )
        else:
            # We are wrapping.  First ensure data is cached
            _ = self._old_tod.local_velocity()
            # Now return a reference to the cached data
            cachename = self._keynames["velocity"]
            return self._old_tod.cache.reference(cachename)

    def position(self, name=None):
        if self._old_tod is None:
            raise NotImplementedError(
                "The Observation class currently only wraps a legacy TOD"
            )
        else:
            # We are wrapping.  First ensure data is cached
            _ = self._old_tod.local_position()
            # Now return a reference to the cached data
            cachename = self._keynames["position"]
            return self._old_tod.cache.reference(cachename)

    def common_flags(self, name=None):
        if self._old_tod is None:
            raise NotImplementedError(
                "The Observation class currently only wraps a legacy TOD"
            )
        else:
            # We are wrapping.  First ensure data is cached
            _ = self._old_tod.local_common_flags()
            # Now return a reference to the cached data
            cachename = self._keynames["common_flags"]
            return self._old_tod.cache.reference(cachename)

    def hwp_angle(self, name=None):
        if self._old_tod is None:
            raise NotImplementedError(
                "The Observation class currently only wraps a legacy TOD"
            )
        else:
            # We are wrapping.  First ensure data is cached
            _ = self._old_tod.local_hwp_angle()
            # Now return a reference to the cached data
            cachename = self._keynames["hwp_angle"]
            return self._old_tod.cache.reference(cachename)

    def boresight_azel(self, name=None):
        if self._old_tod is None:
            raise NotImplementedError(
                "The Observation class currently only wraps a legacy TOD"
            )
        else:
            # Return a reference to the cached data
            cachename = self._keynames["boresight_azel"]
            return self._old_tod.cache.reference(cachename)

    def boresight_radec(self, name=None):
        if self._old_tod is None:
            raise NotImplementedError(
                "The Observation class currently only wraps a legacy TOD"
            )
        else:
            # Return a reference to the cached data
            cachename = self._keynames["boresight_radec"]
            return self._old_tod.cache.reference(cachename)

    def signal(self, name=None):
        if self._old_tod is None:
            # Return the Cache instance for this flavor
            raise NotImplementedError(
                "The Observation class currently only wraps a legacy TOD"
            )
        else:
            # We are wrapping.  First ensure data is cached, then return a dict of
            # references.
            refs = dict()
            for d in self._old_tod.local_dets:
                r = self._old_tod.local_signal(d)
                del r
                cachename = "{}_{}".format(self._keynames["signal"], d)
                refs[d] = self._old_tod.cache.reference(cachename)
            return refs

    def flags(self, name=None):
        if self._old_tod is None:
            # Return the Cache instance for this flavor
            raise NotImplementedError(
                "The Observation class currently only wraps a legacy TOD"
            )
        else:
            # We are wrapping.  First ensure data is cached, then return a dict of
            # references.
            refs = dict()
            for d in self._old_tod.local_dets:
                r = self._old_tod.local_flags(d)
                del r
                cachename = "{}_{}".format(self._keynames["flags"], d)
                refs[d] = self._old_tod.cache.reference(cachename)
            return refs

    # def response(self, name=None): # Instrument common Mueller matrix.
    # def velocity(self, name=None): # Telescope velocity.
    # def position(self, name=None): # Telescope position.

    def __getitem__(self, key):
        if (key == "tod") and (self._old_tod is not None):
            return self._old_tod
        else:
            return self._internal[key]

    def __delitem__(self, key):
        del self._internal[key]

    def __setitem__(self, key, value):
        self._internal[key] = value

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def __repr__(self):
        val = "<Observation \n"
        if self._mpicomm is None:
            val += "  group has a single process (no MPI)"
        else:
            val += "  group has {} processes".format(self._mpicomm.size)
        val += "\n  telescope = {}".format(self._telescope.__repr__())
        for k, v in self._internal.items():
            val += "\n  {} = {}".format(k, v)
        val += "\n>"
        return val


def wrap_old(old_obs):
    """Construct a new Observation that wraps an old observation dictionary / TOD.

    Args:
        old_obs (dict):  The old observation dictionary.

    Returns:
        (Observation):  A new Observation class wrapping the old data.

    """
    import traceback

    # First get the TOD
    old_tod = old_obs["tod"]

    # Get the keynames
    keynames = {
        "times": old_tod.TIMESTAMP_NAME,
        "common": "common",
        "common_flags": old_tod.COMMON_FLAG_NAME,
        "velocity": old_tod.VELOCITY_NAME,
        "position": old_tod.POSITION_NAME,
        "signal": old_tod.SIGNAL_NAME,
        "flags": old_tod.FLAG_NAME,
        "boresight_azel": "boresight_azel",
        "boresight_radec": "boresight_radec",
        "hwp_angle": old_tod.HWP_ANGLE_NAME,
    }

    # Compute the nominal effective sample rate
    rate = None
    if (old_tod.grid_comm_col is None) or (old_tod.grid_comm_col.rank == 0):
        # Only the first process row needs to do this
        stamps = old_tod.local_times()
        rate = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
    if old_tod.grid_comm_row is not None:
        rate = old_tod.grid_comm_row.reduce(rate, op=MPI.SUM, root=0)
        rate /= old_tod.grid_comm_row.size
    if old_tod.mpicomm is not None:
        rate = old_tod.mpicomm.bcast(rate, root=0)

    # And the noise properties
    old_noise = None
    if "noise" in old_obs:
        old_noise = old_obs["noise"]

    # Now construct a telescope with a focalplane

    det_data = dict()
    detnames = old_tod.detectors
    detindx = old_tod.detindx
    detoffset = old_tod.detoffset()
    for d in detnames:
        props = dict()
        props["UID"] = detindx[d]
        props["index"] = detindx[d]
        props["quat"] = detoffset[d]
        if old_noise is not None:
            # We have a noise model.  If its an analytic model, we can copy
            # the parameters over for convenience
            try:
                props["fmin"] = old_noise.fmin(d)
                props["fknee"] = old_noise.fknee(d)
                props["alpha"] = old_noise.alpha(d)
                props["NET"] = old_noise.NET(d)
            except:
                pass
        det_data[d] = props

    is_ground = False
    try:
        _ = old_tod.read_boresight_azel(local_start=0, n=1)
        is_ground = True
    except NotImplementedError:
        pass
    coord = "E"
    if is_ground:
        coord = "C"

    fp = Focalplane(sample_rate=rate, detector_data=det_data)

    tele = Telescope(
        old_obs["telescope"],
        id=old_obs["telescope_id"],
        focalplane=fp,
        site=old_obs["site"],
        coord=coord,
    )

    # Create the Observation
    obs = Observation(tele, old_tod=old_tod, keynames=keynames)

    # Copy over any other metadata
    obs["noise"] = old_noise

    ignore = ["tod", "noise", "baselines", "intervals"]
    for k, v in old_obs.items():
        if k not in ignore:
            obs[k] = v

    return obs
