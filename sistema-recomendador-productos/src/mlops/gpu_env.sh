#!/bin/bash
export LD_LIBRARY_PATH=$(find /root/gpt-gpu/lib/python3.10/site-packages/nvidia -type d -name lib | tr '\n' ':')$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=2
exec /root/gpt-gpu/bin/python "$@"
