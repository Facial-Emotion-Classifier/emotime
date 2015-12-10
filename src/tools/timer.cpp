//#include <sys/time.h>
#include <time.h>
#include "timer.h"
#include <ctime>

double timestamp()
{
	return time(0);
}