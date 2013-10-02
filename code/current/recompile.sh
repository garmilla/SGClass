#! /bin/sh

rm *so
rm -rf dist/
python setup.py bdist
cd dist
tar -xzvf UNKNOWN-0.0.0.linux-x86_64.tar.gz
cd u/garmilla/EPD/epd-7.3-2-rh5-x86_64/lib/python2.7/site-packages/ 
cp *so ~/Source/star-galaxy-classification/code/current/
