import numpy as np
import lbl


for w in np.array([50,100,200,500]):
    lbl.compute(config_file = '/Users/eartigau/lbl/config.yaml',lblrv_subdir = 'lblrv_w'+str(w),hp_width = w,
                skip_done = True)
    #all_tbl = lbl.compil(config_file = '/Users/eartigau/lbl/config.yaml',lblrv_subdir = 'lblrv_w'+str(w))
    #tbl=all_tbl['rdb_table']