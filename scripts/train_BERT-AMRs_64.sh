#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

(
    python main.py -m train --data amr -s -e 10 -l 2e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 5 -l 2e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 3 -l 2e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 10 -l 3e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 5 -l 3e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 3 -l 3e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 10 -l 5e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 5 -l 5e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -s -e 3 -l 5e-5 -b 64 -p
) &
