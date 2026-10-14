# Generated from packaging/homebrew/drama-rs.rb in apiplant/drama-rs by the
# release workflow, which fills in the version and checksum and commits the
# result to apiplant/homebrew-tap as Formula/drama-rs.rb. Changes belong in
# the source repository: the next release overwrites this file.
class DramaRs < Formula
  desc "Fish Audio Drama 3 text-to-speech on a single 24 GB GPU (CUDA)"
  homepage "https://github.com/apiplant/drama-rs"
  version "@VERSION@"
  license "Apache-2.0"

  # Linux x86_64 only: the engine is CUDA. It needs an NVIDIA driver and the
  # CUDA 12 runtime (libcudart, libcublas) on the host, which Homebrew cannot
  # provide.
  depends_on :linux
  depends_on arch: :x86_64

  url "https://github.com/apiplant/drama-rs/releases/download/v@VERSION@/drama-rs-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
  sha256 "@SHA_LINUX_X86_64@"

  def install
    bin.install "drama"
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/drama --version")
  end
end
