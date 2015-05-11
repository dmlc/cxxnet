## Debug the performance

Normally, the GPU ultilizaiton with be above 95% during running. We can get this
number by `nvdia-smi` for high-end Nvidia GPUs, or comparing the required FLOPS of the
neural network against the theoretical capacity of the GPUs.

However, sometimes we didn't get the desired performance. Here we list some
common problems:

1. Check if there is only your program using that GPU cards. If there are
   multiple GPUs cards, you can use `gpu:2` to select the 3rd card.

2. Check if reading the data is the bottleneck. You can test the reading and
   decoding performance by adding `test_io = 1` in your configuration. To improve the
   performance, you can
   - use `iter = threadbuffer` to do data prefetching
   - use a compact binary data format
   - change from `iter = imbin` to `iter = imbinx` to use the multithread
     decoder
   - copy the data into local disk if it sits on a NFS.

3. Use a proper minibatch size. A larger minibatch size improve the system
   performance. But it requires more memory (`~= model_size + minibatch_size *
   const`) and may slows down the convergence. You need to do a trade-off here.

4. Check if the memory-to-GPU bandwidth is the bottleneck. It often happens when
   using multi-GPUs within a single machine. There are several tools to monitor
   the memory bandwidth, such as
   [intel-perf-counter](https://software.intel.com/en-us/articles/intel-performance-counter-monitor)

5. Check if the network bandwidth is the bottleneck for distributed training
   using multiple machines. It often hits the maximal network bandwidth on 1Gbps
   clusters. To reduce the network bandwidth, you can
   1. Increase the minibatch size
   2. Use the filters in parameter server, such as converting a float into a 1
   (or 2) byte integer, and (or) data compression.
