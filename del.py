import os

def delete(folder):
    for file in os.listdir(folder):
        os.remove(folder+"/"+file)

delete("pre")