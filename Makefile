CXX=g++
SDK_PATH= /usr/local/cuda/samples
CUDA_INSTALL_PATH=/usr/local/cuda



CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include -I$(SDK_PATH)/common/inc/
# `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64  -lcudart  `pkg-config --libs opencv` -lcudpp

all: harris main
harris:
	nvcc -O3 -c cuda_kernels/harris_detection.cu -o harris_cuda.o $(CFLAGS)
	nvcc -O3 -c cuda_kernels/get_temp_bytes.cu -o calc_temp_bytes.o $(CFLAGS)
	
main:
	$(CXX) $(CFLAGS) -O3 -c main.cpp -o main.o
	
	$(CXX) $(CFLAGS) -O3 main.o harris_cuda.o calc_temp_bytes.o -o binary.bin $(LDFLAGS)  
	
clean:
	rm -rf *.o *.bin