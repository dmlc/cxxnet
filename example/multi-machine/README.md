
* Build by using `build_ps.sh` in the root directory. All parameter server and
  its dependencies will be statically linked into the binary.

* Generate `train.lst` by following instructions  in [kaggle_bowl](../kaggle_bowl).

* Partition the data into 8 parts.

```
./partition.sh train.lst ../../tools/im2bin
```

* Assume there are two machines, and their IPs are saved in `hosts`
```bash
$ cat hosts
192.168.0.111
192.168.0.112
```
Further assume each machine has two GPUs, so we put `dev = gpu:0,1` in
`bowl.conf`. If `mpirun` is installed,  then launch `cxxnet` on these two
machines by using 2 workers and 2 servers:
```
./run.sh 2 2 bowl.conf
```

More advantaged usage:

 - put all log files in ./log
   ```
   ./run.sh 2 2 bowl.conf -log_dir log
   ```
 - log all network package information (namely enable all verbose in `system/van.h` and `system/van.cc`)
   ```
   ./run.sh 2 2 bowl.conf -vmodule van*=1
   ```

## TODO

* Data partition is not necessary when dmlc-core is ready.
* The distributed version doesn't support `xavier` initialization. The temp
  solution is using `convert.py`. (The current version in this directory is
  buggy, waiting for Bing's patch).
* Rather than let the root node do the evaluation, do it in the distributed
  fashion.
* A distributed monitor for better progress printing.
* More testing
