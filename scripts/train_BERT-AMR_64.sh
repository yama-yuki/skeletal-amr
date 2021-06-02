#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

(
    python main.py -m train --data amr -e 10 -l 2e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -e 5 -l 2e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -e 3 -l 2e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -e 10 -l 3e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -e 5 -l 3e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -e 3 -l 3e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -e 10 -l 5e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -e 5 -l 5e-5 -b 64 -p
) &
wait $!
(
    python main.py -m train --data amr -e 3 -l 5e-5 -b 64 -p
) &

