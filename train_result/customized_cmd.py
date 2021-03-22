import os
from train_result.config import task_folder, batch_folder

with open("{}.txt".format(os.path.join(task_folder, batch_folder, batch_folder)), "r") as f:
    lines = [line for line in f.readlines() if line != "\n"]

train_begin, train_end = 24,24
CUDA = '3'
target_cmds = lines[ train_begin: ]
# target_cmds = [cmd for cmd in target_cmds if cmd != ""]

if CUDA != -1:
    cmds = [cmd[:22] + str(CUDA) + cmd[22:-1] + ",\n" for cmd in target_cmds]
else:
    cmds = [cmd[0] + cmd[23:-1] + ",\n" for cmd in target_cmds]

with open("{}/cmds.txt".format(os.path.join(task_folder, batch_folder)), "a+") as cf:
    for cmd in cmds:
        cf.write(cmd)
    cf.write("\n")
