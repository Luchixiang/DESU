import os

os.system("rm -r data")
if not os.path.isdir('./data'):
    os.mkdir('./data')

id='11aflcrcatFRkv7EabjWjdlpT0DYRbUDZ'
command="wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=11aflcrcatFRkv7EabjWjdlpT0DYRbUDZ\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id=11aflcrcatFRkv7EabjWjdlpT0DYRbUDZ\" -O data/out.tar && rm -rf /tmp/cookies.txt"
print (os.system(command) )
import tarfile,sys

tar = tarfile.open('data/out.tar')
tar.extractall('data')
tar.close()