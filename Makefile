NVCC = nvcc

# -arch=sm_87 is specifically for the Jetson AGX Orin Ampere architecture
NVCC_FLAGS = -O3 -arch=sm_87 \
             -Iinclude \
             -Ilibsmctrl \
             -I/usr/local/cuda-11.4/samples/3_Imaging/stereoDisparity

LDFLAGS = -Llibsmctrl -lsmctrl -lcuda -lcudart

SRCS = main.cu \
       src/victims/matrixMul.cu \
       src/victims/vectorAdd.cu \
       src/victims/stereoDisparity.cu \
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
