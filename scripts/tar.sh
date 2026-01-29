#!/bin/sh

# tar --exclude=usparse/docs/ -zcvf usparse.tar.gz usparse 

gtar -zcvf usparse.tar.gz --exclude usparse/docs  usparse

echo "created usparse.tar.gz"
