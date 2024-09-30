CC=clang++
CFLAGS+= -Os -I/opt/cuda/include -I$(shell dirname $(shell which $(CC)))/../include -fno-rtti -I../doomgeneric -D_REENTRANT -g
LDFLAGS+=-Wl,--gc-sections -L/opt/cuda/lib -L$(shell dirname $(shell which $(CC)))/../lib
LIBS+=-lllvmlibc_rpc_server -lcuda -lLLVMSupport -lLLVMDemangle -lLLVMObject -lLLVMTargetParser -ltinfo -lSDL2

# subdirectory for objects
OBJDIR=build
OUTPUT=nvidia-loader

SRC_DOOM = nvidia-loader.o
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

