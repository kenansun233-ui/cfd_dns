NVCC ?= nvcc

TARGET ?= dns_solver
SRC_DIR := dns_maincode
INC_DIR := src
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
OUTPUT_DIR := OutputFile
RUN_DIR ?= default

ARCH ?= sm_75
NVCCFLAGS ?= -O3 -std=c++14 -arch=$(ARCH) -I$(INC_DIR)
LDFLAGS ?= -lcufft

CU_SOURCES := \
	$(SRC_DIR)/main.cu \
	$(SRC_DIR)/init.cu \
	$(SRC_DIR)/rhs.cu \
	$(SRC_DIR)/info_device.cu

CPP_SOURCES := \
	$(SRC_DIR)/parameters.cpp

OBJECTS := \
	$(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SOURCES)) \
	$(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))

HEADERS := $(wildcard $(INC_DIR)/*.h) $(wildcard $(INC_DIR)/*.cuh)

.PHONY: all run debug gpuinfo clean clean-output clean-case cleanall dirs

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

run: $(TARGET) | $(OUTPUT_DIR)
	./$(TARGET) $(RUN_DIR)

debug:
	@echo "TARGET=$(TARGET)"
	@echo "ARCH=$(ARCH)"
	@echo "RUN_DIR=$(RUN_DIR)"
	@echo "OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "NVCCFLAGS=$(NVCCFLAGS)"

gpuinfo:
	nvidia-smi

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

clean-output:
	rm -rf $(OUTPUT_DIR)

clean-case:
	@test -n "$(RUN_DIR)"
	@case "$(RUN_DIR)" in /*|*..*) echo "Invalid RUN_DIR=$(RUN_DIR)"; exit 1;; esac
	rm -rf "$(OUTPUT_DIR)/$(RUN_DIR)"

cleanall: clean clean-output
