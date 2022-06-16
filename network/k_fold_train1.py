import os

for i in range(5):
    cmd = 'python OriginalVAE.py --cuda --nepoch 20'
    os.system(cmd)
print("Train Original ok!")