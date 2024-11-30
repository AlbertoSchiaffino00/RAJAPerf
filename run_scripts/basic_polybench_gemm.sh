
# sizes = [64, 256, 1024, 4096, 16384, 65536]
sizes="64 256 1024 4096 16384 65536"

for size in $sizes
do
  ./raja-perf-omptarget.exe -k Polybench_GEMM --size $size --repfact 5.0 --disable-warmup -v Base_Seq Base_OMPTarget RAJA_OMPTarget -od outputs/$size --omptarget-data-space Host
done