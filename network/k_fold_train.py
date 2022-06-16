import os

for i in range(8):
    cmd = ' python classify.py --cuda --encoder 1 --outf /home/njuciairs/zmy --img_interval 100'
    os.system(cmd)
print("Train classify ok!")