
    
import sys
import os

sys.path.append('shapeset2_1cspo_2_3.5000.test_code')

from image_config import *

d = dataset(shapeset=2,seed=2,tag='test',n_examples=5000,free='color,size,position,orientation')
d.data_directory = "."

try:
    task = sys.argv[1]
    args = sys.argv[2:]

    fn = getattr(d, task)
    
except Exception, e:
    print 'Usage: %(prog)s write_formats [format]*\n       %(prog)s view' % {'prog': sys.argv[0]}
    sys.exit()

fn(*args)

