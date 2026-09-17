NVCC = nvcc

# -arch=sm_87 is specifically for the Jetson AGX Orin Ampere architecture
NVCC_FLAGS = -O3 -arch=sm_87 \
             -DGPU \
             -Iinclude \
             -Ilibsmctrl \
             -Isrc/victims \
             -Isrc/victims/HSOpticalFlow \
             -Isrc/victims/darknet/include \
             -Isrc/victims/darknet/src \
             -I/usr/local/cuda/include \
             -I/usr/local/cuda-11.4/samples/common/inc \
             -I/usr/local/cuda-11.4/samples/3_Imaging/stereoDisparity

LDFLAGS = -Llibsmctrl -lsmctrl \
          -Lsrc/victims/darknet -ldarknet \
          -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand \
          -lm -lpthread -lstdc++

SRCS = main.cu \
       src/victims/matrixMul.cu \
       src/victims/vectorAdd.cu \
       src/victims/stereoDisparity.cu \
       src/victims/hsOpticalFlow.cu \
       src/victims/HSOpticalFlow/flowCUDA.cu \
       src/victims/yolo.cu \
       src/enemies/stress_compute.cu \
       src/enemies/stress_memory.cu

OBJS = $(SRCS:.cu=.o)
TARGET = profiler

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)