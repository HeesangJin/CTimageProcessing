#include <cstdio>

int main(void)
{
    FILE    *fp_sour;
    FILE    *fp_dest;
    char     buff[1024];
    size_t   n_size;
    
    fp_sour  = fopen("./main.c"  , "r");
    fp_dest  = fopen("./main.bck", "w");
    
    while( 0 < (n_size = fread(buff, 1, 1024, fp_sour)))
    {
        fwrite(buff, 1, n_size, fp_dest);
    }
    
    fclose(fp_sour);
    fclose(fp_dest);
    return 0;
}