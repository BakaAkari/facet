"""Hold N GiB of touched RAM until a sentinel file appears (SC7 only: makes a baseline scan evict).

usage: ballast.py GIB SENTINEL_PATH [--timeout SECONDS]

The parent starts this as a child process:
    python ballast.py 8 D:\\facet-scratch\\perf2\\ballast.stop &
and ends it by creating the sentinel file (or lets --timeout, default 1800 s, end it). Memory is
allocated in 64 MiB blocks, every page is written so it is committed (not just reserved), and the
process prints how much it actually holds. Exit 0 when released by the sentinel, 3 on timeout,
2 on a usage error. Nothing is written outside the sentinel's directory.
"""
import os
import sys
import time

BLOCK = 64 * 2 ** 20


def main(argv):
    timeout = 1800.0
    if '--timeout' in argv:
        i = argv.index('--timeout')
        timeout = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    gib = float(argv[0])
    sentinel = argv[1]
    if gib <= 0:
        print('GIB must be positive')
        return 2
    if os.path.exists(sentinel):
        print('sentinel already exists, refusing to start: %s' % sentinel)
        return 2
    held = []
    want = int(gib * 2 ** 30)
    got = 0
    while got < want:
        n = min(BLOCK, want - got)
        b = bytearray(n)
        for off in range(0, n, 4096):   # touch every page so it is committed
            b[off] = 1
        held.append(b)
        got += n
    print('ballast: holding %.2f GiB (pid %d); waiting for %s' % (got / 2 ** 30, os.getpid(), sentinel), flush=True)
    t0 = time.time()
    while not os.path.exists(sentinel):
        if time.time() - t0 > timeout:
            print('ballast: timeout after %.0fs, releasing' % timeout, flush=True)
            return 3
        time.sleep(0.5)
    print('ballast: sentinel seen, releasing', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
