.. _changes:

Change Log
-------------------------

2.3.9 (Unreleased)
~~~~~~~~~~~~~~~~~~~~~~~~~

* No changes yet

2.3.8 (2020-06-27)
~~~~~~~~~~~~~~~~~~~~~~~~~

* Minor release focusing on build system changes to support packaging
* Update bundled pybind11 and other fixes for wheels and conda packages (PR `#359`_).

.. _`#359`: https://github.com/hpc4cmb/toast/pull/359

2.3.7 (2020-06-13)
~~~~~~~~~~~~~~~~~~~~~~~~~

* Documentation updates and deployment of pip wheels on tags (PR `#356`_).
* Cleanups of conviqt polarization support (PR `#347`_).
* Support elevation nods in ground simulations (PR `#355`_).
* Fix a bug in parallel writing of Healpix FITS files (PR `#354`_).
* Remove dependence on MPI compilers.  Only mpi4py is needed (PR `#350`_).
* Use the native mapmaker by default in the example pipelines (PR `#352`_).
* Updates to build system for pip wheel compatibility (PR `#348`_, `#351`_).
* Switch to github actions instead of travis for continuous integration (PR `#349`_).
* Updates to many parts of the simulation and filtering operators (PR `#341`_).
* In the default Healpix pointing matrix, support None for HWP angle (PR `#345`_).
* Add support for HWP in conviqt beam convolution (PR `#343`_).
* Reimplementation of example jobs used for benchmarks (PR `#332`_).
* Apply atmosphere scaling in temperature, not intensity (PR `#328`_).
* Minor bugfix in binner when running in debug mode (PR `#325`_).
* Add optional boresight offset to the scheduler (PR `#329`_).
* Implement helper tools for parsing mapmaker options (PR `#321`_).

.. _`#356`: https://github.com/hpc4cmb/toast/pull/356
.. _`#347`: https://github.com/hpc4cmb/toast/pull/347
.. _`#355`: https://github.com/hpc4cmb/toast/pull/355
.. _`#354`: https://github.com/hpc4cmb/toast/pull/354
.. _`#350`: https://github.com/hpc4cmb/toast/pull/350
.. _`#352`: https://github.com/hpc4cmb/toast/pull/352
.. _`#351`: https://github.com/hpc4cmb/toast/pull/351
.. _`#348`: https://github.com/hpc4cmb/toast/pull/348
.. _`#349`: https://github.com/hpc4cmb/toast/pull/349
.. _`#341`: https://github.com/hpc4cmb/toast/pull/341
.. _`#345`: https://github.com/hpc4cmb/toast/pull/345
.. _`#343`: https://github.com/hpc4cmb/toast/pull/343
.. _`#332`: https://github.com/hpc4cmb/toast/pull/332
.. _`#328`: https://github.com/hpc4cmb/toast/pull/328
.. _`#325`: https://github.com/hpc4cmb/toast/pull/325
.. _`#329`: https://github.com/hpc4cmb/toast/pull/329
.. _`#321`: https://github.com/hpc4cmb/toast/pull/321

2.3.6 (2020-01-19)
~~~~~~~~~~~~~~~~~~~~~~~~~

* Overhaul documentation (PR `#320`_).
* Small typo fix for conviqt operator (PR `#319`_).
* Support high-cadence ground scan strategies and fix a bug in turnaround simulation (PR `#316`_).
* Fix BLAS / LAPACK name mangling detection (PR `#315`_).
* Allow disabling sky sim in example pipeline (PR `#313`_).

.. _`#320`: https://github.com/hpc4cmb/toast/pull/320
.. _`#319`: https://github.com/hpc4cmb/toast/pull/319
.. _`#316`: https://github.com/hpc4cmb/toast/pull/316
.. _`#315`: https://github.com/hpc4cmb/toast/pull/315
.. _`#313`: https://github.com/hpc4cmb/toast/pull/313


2.3.5 (2019-11-19)
~~~~~~~~~~~~~~~~~~~~~~~~~

* Documentation updates (PR `#310`_).

.. _`#310`: https://github.com/hpc4cmb/toast/pull/310


2.3.4 (2019-11-17)
~~~~~~~~~~~~~~~~~~~~~~~~~

* Disabling timing tests during build of conda package.


2.3.3 (2019-11-16)
~~~~~~~~~~~~~~~~~~~~~~~~~

* Change way that the MPI communicator is passed to C++ (PR `#309`_).

.. _`#309`: https://github.com/hpc4cmb/toast/pull/309


2.3.2 (2019-11-13)
~~~~~~~~~~~~~~~~~~~~~~~~~

* Convert atmosphere simulation to new libaatm package (PR `#307`_).
* Improve vector math unit tests (PR `#296`_).
* Updates to conviqt operator (PR `#304`_).
* Satellite example pipeline cleanups.
* Store local pixel information in the data dictionary (PR `#306`_).
* Add elevation-dependent noise (PR `#303`_).
* Move global / local pixel lookup into compiled code (PR `#302`_).
* PySM operator changes to communicator (PR `#301`_).
* Install documentation updates (PR `#300`_, `#299`_).

.. _`#307`: https://github.com/hpc4cmb/toast/pull/307
.. _`#296`: https://github.com/hpc4cmb/toast/pull/296
.. _`#304`: https://github.com/hpc4cmb/toast/pull/304
.. _`#306`: https://github.com/hpc4cmb/toast/pull/306
.. _`#303`: https://github.com/hpc4cmb/toast/pull/303
.. _`#302`: https://github.com/hpc4cmb/toast/pull/302
.. _`#301`: https://github.com/hpc4cmb/toast/pull/301
.. _`#300`: https://github.com/hpc4cmb/toast/pull/300
.. _`#299`: https://github.com/hpc4cmb/toast/pull/299


2.3.1 (2019-10-14)
~~~~~~~~~~~~~~~~~~~~~~~~~

* Fix bug when writing FITS maps serially.
* Improve printing of Environment and TOD classes (PR `#294`_).
* Fix a race condition (PR `#292`_).
* Control the Numba backend from TOAST (PR `#283`_, `#291`_).
* Functional TOAST map-maker (PR `#288`_).
* Large improvements to ground sim example (PR `#290`_).
* Overhaul examples to match 2.3.0 changes (PR `#286`_).
* Handle small angles and improve unit tests for healpix.

.. _`#294`: https://github.com/hpc4cmb/toast/pull/294
.. _`#292`: https://github.com/hpc4cmb/toast/pull/292
.. _`#283`: https://github.com/hpc4cmb/toast/pull/283
.. _`#291`: https://github.com/hpc4cmb/toast/pull/291
.. _`#288`: https://github.com/hpc4cmb/toast/pull/288
.. _`#290`: https://github.com/hpc4cmb/toast/pull/290
.. _`#286`: https://github.com/hpc4cmb/toast/pull/286


2.3.0 (2019-08-13)
~~~~~~~~~~~~~~~~~~~~~~~~~

* Rewrite of internal compiled codebase and build system.
* Move common pipeline configuration to a new module (PR `#280`_).
* Add scan synchronous simulation operator (PR `#278`_).

.. _`#280`: https://github.com/hpc4cmb/toast/pull/280
.. _`#278`: https://github.com/hpc4cmb/toast/pull/278
