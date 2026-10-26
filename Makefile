MODULE_TOPDIR = ../..

PGM = i.delineate.opencl

LIBES = $(IMAGERYLIB) $(VECTORLIB) $(DBMILIB) $(RASTERLIB) $(GISLIB) \
	$(BLASLIB) $(MATHLIB)
DEPENDENCIES = $(IMAGERYDEP) $(VECTORDEP) $(DBMIDEP) $(RASTERDEP) \
	$(GISDEP)
EXTRA_INC = $(VECT_INC) $(OCLINCPATH) $(OPENMP_INCPATH) -I$(OBJDIR)
EXTRA_CFLAGS = $(VECT_CFLAGS) $(OPENMP_CFLAGS) -O3 -std=gnu11
EXTRA_LIBS = $(OCLLIBPATH) $(OCLLIB) $(OPENMP_LIBPATH) $(OPENMP_LIB) -lcurl

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# Embed the OpenCL kernels as a C string literal.
$(OBJDIR)/ocl_backend.o: $(OBJDIR)/yolo_kernels_cl.h

$(OBJDIR)/yolo_kernels_cl.h: yolo_kernels.cl | $(OBJDIR)
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< > $@
