CC = gcc
LD = ${CC}

OBJ = main.o bbfmm.o \
	options.o \
	checks.o \
	opts.o \
	envs.o \
	output.o \
	timer.o \
	anotherFMMCompute.o \
	anotherFMMInteraction.o \
	anotherDownwardPassX.o \
	anotherDownwardPass.o \
	anotherUpwardPass.o \
	anotherFMMDistribute.o \
	anotherInteractionList.o \
	anotherNearField.o \
	m2l_aux.o \
	auxAnotherFMMInteraction.o \
	anotherBuildFMMHierachy.o \
	xdot.o atransx.o \
	m2l_aux_cpu.o

NOOPTOBJ = elapsed.o

OPTS = -std=c99 -O3 -Wall -DSINGLE

KERNEL = -DLAPLACIAN  # use -DLAPLACIAN, -DLAPLACIANFORCE, or -DONEOVERR4
OPTS += ${KERNEL}

INCLUDES =
LIBS = 

CFLAGS = ${OPTS} ${INCLUDES}

LDFLAGS = -L/home/ttaka/lib -llapack${PLAT} -lblas${PLAT} -lm

EXE = a.out

TAR = tmp.tar

${EXE}: ${OBJ} ${NOOPTOBJ} bbfmm.h another.h
	${CC} ${OPTS} -o ${EXE} ${OBJ} elapsed.o ${LDFLAGS} ${INCLUDES} ${LIBS} ${OMP}
${OBJ}:
elapsed.o: elapsed.c
	${CC} -O0 ${OPTS_ELAPSED} -o elapsed.o -c elapsed.c -DCPU_CLOCK_GHZ=${CPU_CLOCK_GHZ}
clean : 
	rm -f *.o core
realclean:
	rm -f *.o ${CUEXE}
tar backup:
	(tar cvf ${TAR} COPYING README Makefile Makefile.cuda *.sh *.h *.c *.cu work/*.{sh,c})
	gzip -f ${TAR}
#diff:
#	(./diff.sh ../111213/ ${OBJ:.o=.c} ${NOOPTOBJ:.o=.c} *.cu *.h)
