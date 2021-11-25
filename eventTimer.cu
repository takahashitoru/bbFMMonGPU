#ifndef EVENTTIMER_CU
#define EVENTTIMER_CU

#include "eventTimer.h"

void initEventTimer(eventTimerType *timer)
{
  CSC(cudaEventCreate(&(timer->start)));
  CSC(cudaEventCreate(&(timer->stop)));
  timer->time = 0.0f;
}  

void startEventTimer(eventTimerType *timer)
{
  CSC(cudaEventRecord(timer->start, 0));
}

void stopEventTimer(eventTimerType *timer)
{
  CSC(cudaEventRecord(timer->stop, 0));
  CSC(cudaEventSynchronize(timer->stop));
  float elapsedTime;
  CSC(cudaEventElapsedTime(&elapsedTime, timer->start, timer->stop));
  timer->time += elapsedTime;
}

void finalizeEventTimer(eventTimerType *timer)
{
  CSC(cudaEventDestroy(timer->start));
  CSC(cudaEventDestroy(timer->stop));
}

void printEventTimer(FILE *fp, char *s, eventTimerType *timer)
{
#if defined(ENABLE_PRINT_TIMER_IN_F_FORMAT)
  fprintf(fp, "# %s: %s = %f\n", __FUNCTION__, s, timer->time / 1000); /* milisec -> sec */
#else
  fprintf(fp, "# %s: %s = %14.7e\n", __FUNCTION__, s, timer->time / 1000); /* milisec -> sec */
#endif
}

double readEventTimer(eventTimerType *timer)
{
  return (double)timer->time; // milisec
}

double getEventTimer(eventTimerType *timer)
{
  return (double)(timer->time / 1000); // milisec -> sec
}

void setEventTimer(eventTimerType *timer, double val)
{
  timer->time = (float)val; // milisec
}  

#endif /* EVENTTIMER_CU */
