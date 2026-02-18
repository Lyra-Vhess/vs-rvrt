# Maintainer: Your Name <your.email@example.com>
pkgname=vsrvrt
pkgver=1.0.0
pkgrel=1
pkgdesc="Vapoursynth plugin for RVRT (Recurrent Video Restoration Transformer) video restoration"
arch=('x86_64')
url="https://github.com/Lyra-Vhess/vs-rvrt"
license=('CC-BY-NC')
depends=(
    'python'
    'python-einops'
    'python-torchvision'
    'python-numpy'
    'python-requests'
    'python-tqdm'
    'vapoursynth'
)
optdepends=(
    'cuda: For GPU acceleration'
    'ffmpeg: For video encoding/decoding'
)
makedepends=('python-setuptools' 'python-wheel')

source=()
md5sums=()

build() {
    # Work directly from the source directory
    cd ../vsrvrt
    python setup.py build
}

package() {
    # Work directly from the source directory
    cd ../vsrvrt
    python setup.py install --root="$pkgdir" --optimize=1
    
    # Install license
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
    
    # Install documentation
    install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
}
