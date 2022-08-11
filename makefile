# Location of the CUDA Toolkit，默认的路径即可
CUDA_PATH ?= "/usr/local/cuda"


# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
	ifneq ($(TARGET_ARCH),$(HOST_ARCH))
		ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
			TARGET_SIZE := 64
		else ifneq (,$(filter $(TARGET_ARCH),armv7l))
			TARGET_SIZE := 32
		endif
	else
		TARGET_SIZE := $(shell getconf LONG_BIT)
	endif
else
	$(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
		$(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
	endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
	TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
	$(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE} -g -G
CCFLAGS     := -g
LDFLAGS     :=


ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
		ifneq ($(TARGET_FS),)
			GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
			ifeq ($(GCCVERSIONLTEQ46),1)
				CCFLAGS += --sysroot=$(TARGET_FS)
			endif
			LDFLAGS += --sysroot=$(TARGET_FS)
			LDFLAGS += -rpath-link=$(TARGET_FS)/lib
			LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
			LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
		endif
	endif
endif

# Debug build flags
ifeq ($(dbg),1)
	  NVCCFLAGS += -g -G
	  BUILD_TYPE := debug
else
	  BUILD_TYPE := release
endif

#　这里添加编译参数，比如-keep， -Xptxas 等等
ALL_CCFLAGS := -Xptxas -v
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I/home/yhao/d/cuda-samples/Common
LIBRARIES :=

################################################################################

# Gencode arguments
# 这里写你的GPU计算能力
SMS ?= 80

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
all: build

# 
build: test

test.o: test.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

test: test.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./test

clean:
	rm -f test test.o

clobber: clean