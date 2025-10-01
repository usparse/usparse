CXX = hipcc

CXXFLAGS = -O2 -std=c++14 -w
LDFLAGS  =

DIR_OBJ = obj
DIR_SRC = src
DIR_INC = inc

CPP_SRCS := $(shell find $(DIR_SRC) -name '*.cpp')

AOBJS := $(patsubst $(DIR_SRC)/%.cpp,$(DIR_OBJ)/%.o,$(CPP_SRCS))

DEPS := $(shell find $(DIR_INC) -name '*.hpp') \
        $(shell find $(DIR_SRC) -name '*.h') \
        $(shell find $(DIR_SRC) -name '*.hpp')

INCLUDES := $(addprefix -I,$(shell find $(DIR_INC) -type d)) \
            $(addprefix -I,$(shell find $(DIR_SRC) -type d))

LIBS = -L$(ROCM_PATH)/lib64 -lrocsparse -lamdhip64 -lhipsparse -lhipblas

PROG = gmres

$(DIR_OBJ)/%.o: $(DIR_SRC)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

$(PROG): $(AOBJS)
	$(CXX) $(LDFLAGS) $(LIBS) $(INCLUDES) $^ -o $@

.PHONY: clean
clean:
	rm -rf $(DIR_OBJ) $(PROG)
