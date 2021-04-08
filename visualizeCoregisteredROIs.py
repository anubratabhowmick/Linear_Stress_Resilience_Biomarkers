import os
import numpy as np
import nibabel as nb
import pathlib
import shutil
import time
from nilearn import plotting

if __name__ == "__main__":
    # Root directory for all files
    rootDirPath = pathlib.Path(r'A:\data\bids')

    # Subject List for the project
    subjectList = ["sub-resi001", "sub-resi002", "sub-resi003", "sub-resi004",
                   "sub-resi005", "sub-resi006", "sub-resi007", "sub-resi008",
                   "sub-resi009", "sub-resi010", "sub-resi011", "sub-resi012",
                   "sub-resi013", "sub-resi014", "sub-resi015", "sub-resi016",
                   "sub-resi017", "sub-resi018", "sub-resi019", "sub-resi020",
                   "sub-resi021", "sub-resi022", "sub-resi023", "sub-resi024",
                   "sub-resi025", "sub-resi026", "sub-resi027", "sub-resi028",
                   "sub-resi029", "sub-resi030", "sub-resi031", "sub-resi032",
                   "sub-resi033", "sub-resi034", "sub-resi035", "sub-resi036",
                   "sub-resi037", "sub-resi038", "sub-resi039", "sub-resi040",
                   "sub-resi041", "sub-resi042", "sub-resi043", "sub-resi044",
                   "sub-resi045", "sub-resi046", "sub-resi047", "sub-resi048",
                   "sub-resi049", "sub-resi050", "sub-resi051", "sub-resi052",
                   "sub-resi053", "sub-resi054", "sub-resi055", "sub-resi056",
                   "sub-resi057", "sub-resi058", "sub-resi059", "sub-resi060",
                   "sub-resi061", "sub-resi062", "sub-resi063", "sub-resi064",
                   "sub-resi065", "sub-resi066", "sub-resi067", "sub-resi068",
                   "sub-resi069", "sub-resi070", "sub-resi071", "sub-resi072",
                   "sub-resi073", "sub-resi074", "sub-resi075", "sub-resi076",
                   "sub-resi077", "sub-resi078", "sub-resi079", "sub-resi080",
                   "sub-resi081", "sub-resi082", "sub-resi083", "sub-resi084",
                   "sub-resi085", "sub-resi086", "sub-resi087", "sub-resi088",
                   "sub-resi089", "sub-resi090", "sub-resi091", "sub-resi092",
                   "sub-resi093", "sub-resi094", "sub-resi095", "sub-resi096",
                   "sub-resi097", "sub-resi098", "sub-resi099", "sub-resi100",
                   "sub-resi101", "sub-resi102", "sub-resi103", "sub-resi104",
                   "sub-resi105", "sub-resi106", "sub-resi107", "sub-resi108",
                   "sub-resi109", "sub-resi110", "sub-resi111", "sub-resi112",
                   "sub-resi113", "sub-resi114", "sub-resi115", "sub-resi116",
                   "sub-resi117", "sub-resi118", "sub-resi119", "sub-resi120",
                   "sub-resi121", "sub-resi122", "sub-resi123", "sub-resi124",
                   "sub-resi125", "sub-resi126", "sub-resi127", "sub-resi128",
                   "sub-resi129", "sub-resi130", "sub-resi131", "sub-resi132",
                   "sub-resi133", "sub-resi134", "sub-resi135", "sub-resi136",
                   "sub-resi137", "sub-resi138", "sub-resi139", "sub-resi140",
                   "sub-resi141", "sub-resi142", "sub-resi143", "sub-resi144",
                   "sub-resi145", "sub-resi146", "sub-resi147", "sub-resi148",
                   "sub-resi149", "sub-resi150"
                   ]

    for subject in subjectList:
        mean_func_file_path = pathlib.Path(rootDirPath / 'bids_native_final' / subject / 'ses-1' / 'func'
                                           / ('art_mean_au' + subject + '_ses-1_task-rest_bold.nii'))

        vwbs_file_path = pathlib.Path(rootDirPath / 'bids_native_final' / subject / 'ses-1' / 'anat'
                                      / (subject + '_ses-1_T1w_vwbs_atlas.nii'))

        if mean_func_file_path.exists() and vwbs_file_path.exists():
            # print('Entering for ', mean_func_file_path)

            mean_func_file = nb.load(mean_func_file_path)
            vwbs_file = nb.load(vwbs_file_path)

            plotting.plot_roi(vwbs_file, mean_func_file, title=subject)
            plotting.show()