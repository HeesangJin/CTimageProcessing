0001: 
COMMENT: wrong rotation

#define NUM_PLANES 150
#define EPSILON_D 0.55
#define EPSILON_J -1

#define N_THETA 12
#define N_Z 18
#define LENGTH_L 8

#define VALUE_S 3
#define VALUE_T 4

input file: 100
Processing time: 6709.58s
=========================
0002:
COMMENT: correct rotation

#define NUM_PLANES 150
#define EPSILON_D 0.55
#define EPSILON_J -1

#define N_THETA 10
#define N_Z 8
#define LENGTH_L 4

#define VALUE_S 3
#define VALUE_T 4
Processing time: 327.242s
=========================
0003:
COMMENT: wrong interoation

#define NUM_PLANES 150
float EPSILON_D = 0.55;
int EPSILON_D_BASE = 0;
float EPSILON_J = -1;
int EPSILON_J_BASE = 0;

#define N_THETA 12
#define N_Z 10
#define LENGTH_L 6

#define VALUE_S 3
#define VALUE_T 4