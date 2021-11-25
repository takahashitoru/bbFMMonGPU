#include "options.h"

void bbfmm_options(int argc, char **argv, int *l, int *seed)
{
  /* Check all the options and arguments */

  {
    int i = 0;
    INFO("%s", argv[i ++]);
    while (i < argc) {
      fprintf(stderr, " %s", argv[i ++]);
    }
    fprintf(stderr, "\n");
  }

  /* Check options */

  int c;

  while ((c = getopt(argc, argv, "d:s:")) != - 1) {
    switch (c) {

    case 'd':

      if (optarg == 0) {
	MESG("Speficify an integer to be added to the number of levels (see FMMsetup). Exit.\n");
	exit(EXIT_FAILURE);
      } else {	
	*l = atoi(optarg);
	INFO("l=%d\n", *l);
      }

      break;

    case 's':

      if (optarg == 0) {
	MESG("Speficify a random seed for generating particles. Exit.\n");
	exit(EXIT_FAILURE);
      } else {	
	*seed = atoi(optarg);
	INFO("seed=%d\n", *seed);
      }

      break;

    default:

      MESG("Invalid option. Exit.\n");
      exit(EXIT_FAILURE);

    }
  }

}
