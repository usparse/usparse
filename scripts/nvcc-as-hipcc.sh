#!/bin/bash

COMPILER_PATH="nvcc"

newcmd="-forward-unknown-to-host-compiler"
flag_xcpp_appear=0

for arg in $@
do
  case $arg in
    "-hc") # remove `-hc`
      ;;
    "-forward-unknown-to-host-compiler")
     # we have set this flag by default, 
     # so we can remove it if it appears in the original command.
      ;;
    *)
      newcmd="$newcmd $arg";;
  esac
done


#if [ "$flag_xcpp_appear" -eq 1 ]; then
#fi
# Finally execute the new command
# add HIPCC macros.
newcmd="-D__HIP_PLATFORM_NVCC__=1 -D__HIPCC__=1 $newcmd "
echo "nvcc $newcmd" >&2
exec $COMPILER_PATH $newcmd
