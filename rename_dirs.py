from glob import glob
import os

directory = '/home/blansdel/projects/brett_dlc/animal_pilot'

dirs = glob(f'{directory}/Social *')

####################
## Rename folders ##
####################

for dr in dirs:
    bn = os.path.basename(dr)
    new_dr = os.path.join(directory, bn.lower().replace(' ', '_'))
    dr_ = dr.replace(' ', '\ ')
    cmd = f"mv {dr_} {new_dr}"
    if not os.path.exists(new_dr):
        print(cmd)
        #os.system(cmd)
    else:
        #os.system(f'rm -r {dr_}')
        print(f'rm -r {dr_}')

########################
## Move overhead vids ##
########################

dirs = glob(f'{directory}/social_*')

dest = '/home/blansdel/projects/brett_dlc/animal_pilot/all_videos'

for dr in dirs:
    file_to_cp = glob(os.path.join(dr, 'e3v813a*'))[0]
    cmd = f'cp {file_to_cp} {dest}'
    #print(cmd)
    os.system(cmd)