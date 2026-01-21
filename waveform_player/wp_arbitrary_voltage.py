
# -*- coding: utf-8 -*-

import time
from sglx_pkg import sglx

# Edit the server address/port here
sglx_addr = "localhost"
sglx_port = 4142

hSglx = sglx.c_sglx_createHandle()
ok    = sglx.c_sglx_connect( hSglx, sglx_addr.encode(), sglx_port )

if ok:
    ip = -1
    slot = 21
    wave_time = 0.05
    n_iter = 300  # number of times to play the whole wave
    import numpy as np
    from numpy import random

    ok = sglx.c_sglx_obx_AO_set( hSglx, ip, slot, '(0,0)'.encode() )
    for i in range(n_iter):
        val = random.uniform(0,5)
        ok = sglx.c_sglx_obx_AO_set( hSglx, ip, slot, f'(0,{val:.2f})'.encode() )
        if ok:                   
            # print(f'iteration: {i}')
            time.sleep(wave_time)
        else:
            break 
    ok = sglx.c_sglx_obx_AO_set( hSglx, ip, slot, '(0,0)'.encode()  )
    
   
if not ok:
    print( "error [{}]\n".format( sglx.c_sglx_getError( hSglx ) ) )

sglx.c_sglx_close( hSglx )
sglx.c_sglx_destroyHandle( hSglx )



