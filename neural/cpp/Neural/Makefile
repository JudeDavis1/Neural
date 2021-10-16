TARGET_EXEC ?= main

BUILD_DIR ?= ./build
SRC_DIRS ?= .

SRCS := $(shell find . -name '*.cc')
#SRCS := $(shell gci -r -fi *.cc)
# INC_DIRS := $(shell find $(SRC_DIRS) -type d)
# INC_FLAGS := $(addprefix -I,includes)

# CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

$(BUILD_DIR)/$(TARGET_EXEC):
	g++ $(SRCS) -o $(BUILD_DIR)/$(TARGET_EXEC) -Iincludes -std=c++17
	./$(BUILD_DIR)/$(TARGET_EXEC)

# .PHONY: clean
# clean:
# 	$(RM) -r $(BUILD_DIR)/$(TARGET_EXEC)

# c++ source
# $(BUILD_DIR)/%.cc.o: %.cc
# 	$(MKDIR_P) $(dir $@)
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o build/main

# -include $(DEPS)

# MKDIR_P ?= mkdir -p
