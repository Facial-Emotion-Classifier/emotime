#include <sys/time.h>
#include <time.h>
#include "timer.h"

double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}