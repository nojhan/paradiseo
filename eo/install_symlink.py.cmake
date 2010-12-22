#!/usr/bin/env python

#NAME    = "@PROJECT_NAME@"
NAME    = "eo"
SOURCE  = "@CMAKE_SOURCE_DIR@"
BINARY  = "@CMAKE_BINARY_DIR@"
PREFIX  = "/usr"

LIBRARIES = ["libcma.a", "libeo.a", "libeoutils.a", "libes.a", "libga.a"]

import os, sys

def isroot():
    if os.getuid() != 0:
        print '[WARNING] you have to be root'
        return False
    return True

def uninstall():
    # remove libraries' symlink
    for lib in LIBRARIES:
        os.remove( "%s/lib/%s" % (PREFIX, lib) )

    # remove headers' symlink
    os.remove( "%s/include/%s" % (PREFIX,NAME) )

    # remove doc's symlink
    os.remove( "%s/share/%s/doc" % (PREFIX, NAME) )

    # remove share directory
    os.rmdir( "%s/share/%s" % (PREFIX, NAME) )

    # remove pkgconfig's symlink
    os.remove( "%s/lib/pkgconfig/%s.pc" % (PREFIX, NAME) )

    print 'All symlinks have been removed.'

def install():
    # create symlink for libraries
    for lib in LIBRARIES:
        os.symlink( "%s/lib/%s" % (BINARY, lib), "%s/lib/%s" % (PREFIX, lib) )

    # create symlink for headers
    os.symlink( "%s/src" % SOURCE, "%s/include/%s" % (PREFIX,NAME) )

    # create share directory
    os.mkdir( "%s/share/%s" % (PREFIX, NAME) )

    # create symlink for doc
    os.symlink( "%s/doc" % BINARY, "%s/share/%s/doc" % (PREFIX, NAME) )

    # create symlink for pkgconfig
    os.symlink( "%s/%s.pc" % (BINARY, NAME), "%s/lib/pkgconfig/%s.pc" % (PREFIX, NAME) )

    print 'All symlinks have been installed.'

if __name__ == '__main__':
    if not isroot():
        sys.exit()

    if len(sys.argv) < 2:
        print 'Usage: %s [install|uninstall]' % sys.argv[0]
        sys.exit()

    if sys.argv[1] == 'install': install()
    elif sys.argv[1] == 'uninstall': uninstall()
