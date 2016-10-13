#!/bin/bash

# #

d=`/bin/pwd`
PYTHONPATH=$d:$d/notebooks/lib:$PYTHONPATH jupyter notebook $*
