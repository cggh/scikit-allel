import sys
sys.path.insert(0, '.')
from allel.io_vcf_read import read_vcf
fn = sys.argv[1]
if len(sys.argv) > 2:
    n_threads = int(sys.argv[2])
else:
    n_threads = None
read_vcf(fn, fields='*', n_threads=n_threads, log=sys.stderr)
