#ifndef EVENTTIMER_H
#define EVENTTIMER_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* Macro for debugging */
#ifdef _DEBUG
#include <cutil.h>
#define CSC(call) CUDA_SAFE_CALL(call)
#else
#define CSC(call) call
#endif

typedef struct {
  float time;
  cudaEvent_t start;
  cudaEvent_t stop;
} eventTimerType;

#ifdef __cplusplus
extern "C" {
#endif
  void initEventTimer(eventTimerType *timer);
  void startEventTimer(eventTimerType *timer);
  void stopEventTimer(eventTimerType *timer);
  void finalizeEventTimer(eventTimerType *timer);
  void printEventTimer(FILE *fp, char *s, eventTimerType *timer);
  double readEventTimer(eventTimerType *timer);
  double getEventTimer(eventTimerType *timer);
  void setEventTimer(eventTimerType *timer, float val);
#ifdef __cplusplus
}
#endif

#endif /* EVENTTIMER_H */
