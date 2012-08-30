# $Id: pkgbuild-mode.el,v 1.23 2007/10/20 16:02:14 juergen Exp $
# Maintainer: Caner Candan <caner@candan.fr>
pkgname=libeo
pkgver=@PROJECT_VERSION@
pkgrel=1
pkgdesc="Evolving Objects is a template-based, ANSI-C++ evolutionary computation library which helps you to write your own stochastic optimization algorithms insanely fast."
url=""
arch=('i686' 'x86_64')
license=('LGPL')
depends=()
makedepends=('make' 'cmake')
conflicts=()
replaces=()
backup=()
install=
source=($pkgname-$pkgver.tar.gz)
md5sums=()
build() {
  cd $startdir/src/$pkgname-$pkgver
  cmake -DCMAKE_INSTALL_PREFIX=/usr .
  make || return 1
  make DESTDIR=$startdir/pkg install
}
