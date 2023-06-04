import gzip
import shutil

with gzip.open('.\\ast-label\\GoogleNews-vectors-negative300.bin.gz', 'rb') as f_in:
    with open('.\\ast-label\\GoogleNews-vectors-negative300.bin', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
