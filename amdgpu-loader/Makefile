CC=clang++
CFLAGS+= -Os -I/opt/rocm/include -I$(shell dirname $(shell which $(CC)))/../include -fno-rtti
LDFLAGS+=-Wl,--gc-sections -L/opt/rocm/lib -L$(shell dirname $(shell which $(CC)))/../lib
LIBS+=-lllvmlibc_rpc_server -lhsa-runtime64 -lLLVMSupport -lLLVMDemangle -ltinfo

# subdirectory for objects
OBJDIR=build
OUTPUT=amdgpu-loader

SRC_DOOM = amdgpu-loader.o
OBJS += $(addprefix $(OBJDIR)/, $(SRC_DOOM))

all: $(OUTPUT)

clean:
	rm -rf $(OBJDIR)
	rm -f $(OUTPUT)

$(OUTPUT):	$(OBJS)
	@echo [Linking $@]
	$(VB)$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) \
	-o $(OUTPUT) $(LIBS)
	@echo [Size]
	-$(CROSS_COMPILE)size $(OUTPUT)

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o:	%.cpp
	@echo [Compiling $<]
	$(VB)$(CC) $(CFLAGS) -c $< -o $@

print:
	@echo OBJS: $(OBJS)

