#!/usr/bin/env python

#NAME    = "@PROJECT_NAME@"
NAME    = "eo"
SOURCE  = "@CMAKE_SOURCE_DIR@"
BINARY  = "@CMAKE_BINARY_DIR@"
PREFIX  = "/usr"

DATA = {
    'dirs': [ "%s/share/%s" % (PREFIX, NAME) ],
    'links': [ ("%s/src" % SOURCE, "%s/include/%s" % (PREFIX, NAME)),
               ("%s/doc" % BINARY, "%s/share/%s/doc" % (PREFIX, NAME)),
               ("%s/%s.pc" % (BINARY, NAME), "%s/lib/pkgconfig/%s.pc" % (PREFIX, NAME)),
	       ]
    }

LIBRARIES = ["libcma.a", "libeo.a", "libeoutils.a", "libes.a", "libga.a"]
DATA['links'] += [ ("%s/lib/%s" % (BINARY, lib), "%s/lib/%s" % (PREFIX, lib)) for lib in LIBRARIES ]

import os, sys

def isroot():
    if os.getuid() != 0:
        print('[WARNING] you have to be root')
        return False
    return True

def uninstall():
    for dummy, link in DATA['links']: os.remove(link)
    for dirname in DATA['dirs']: os.rmdir(dirname)
    print('All symlinks have been removed.')

def install():
    for dirname in DATA['dirs']: os.mkdir(dirname)
    for src, dst in DATA['links']: os.symlink(src, dst)
    print('All symlinks have been installed.')

def data():
    from pprint import pprint
    pprint(DATA, width=200)

if __name__ == '__main__':
    if not isroot():
        sys.exit()

    if len(sys.argv) < 2:
        print(('Usage: %s [install|uninstall|data]' % sys.argv[0]))
        sys.exit()

    if sys.argv[1] == 'install': install()
    elif sys.argv[1] == 'uninstall': uninstall()
    elif sys.argv[1] == 'data': data()
