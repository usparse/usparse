#!/bin/sh

## This script is used for downloading dependencies of usparse.
## Usually, the dependencies is managed by pkg tool (https://github.com/genshen/pkg).
## We can also directly download the dependencies via this script.

## Run this script on the root directory of this project.
mkdir -p vendor/include

# download dependency for cli arguments parsing:
curl https://raw.githubusercontent.com/Taywee/args/refs/tags/6.4.7/args.hxx -o vendor/include/args.hxx
curl https://raw.githubusercontent.com/muellan/clipp/refs/tags/v1.2.3/include/clipp.h -o vendor/include/clipp.h
