"""Process exit codes shared by the CLI jobs and the viewer that spawns them.

Import-free on purpose: the viewer reads these to interpret a ``facet.py`` child's
exit status without importing ``facet`` and its scoring stack.
"""

# A library job refused to start because another job holds the library lock
# (``facet.LibraryLock``). The job is possible, just not now: EX_TEMPFAIL from
# sysexits.h. An unusable lock file is a fault, not a busy library, and exits 1.
LIBRARY_LOCK_HELD_EXIT_CODE = 75
