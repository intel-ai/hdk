#nvcc -O3 -c ../BufferMgr/CudaUtils.cu
g++ --std=c++0x -O3 -I/usr/local/include -o DataMgrTest DataMgrTest.cpp ../DataMgr.cpp ../Encoder.cpp ../../CudaMgr/CudaMgr.cpp ../BufferMgr/BufferMgr.cpp ../BufferMgr/Buffer.cpp ../BufferMgr/CpuBufferMgr/CpuBufferMgr.cpp ../BufferMgr/CpuBufferMgr/CpuBuffer.cpp ../BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.cpp ../BufferMgr/GpuCudaBufferMgr/GpuCudaBuffer.cpp  ../FileMgr/FileMgr.cpp ../FileMgr/FileInfo.cpp ../FileMgr/File.cpp ../FileMgr/FileBuffer.cpp -L/usr/local/lib -lboost_filesystem-mt -lboost_timer-mt -lboost_system-mt -lgtest -L/usr/local/cuda/lib -lcuda -I/usr/local/cuda/include
