#include <cstdio>
#include <cstdint>

int main(void)
{
    FILE    *fp_sour;
    FILE    *fp_dest;
    char     buff[2048];
    size_t   n_size;
    char c;
    
    
    fp_sour  = fopen("./Pramook_black_velvet_3.03um_80kV_down.vol" , "rb");
    n_size = fread( buff, 1, 2048, fp_sour);
//    for(int i=0; i<n_size; i++){
//        printf("%d: %d\n", i+1 ,buff[i]);
//    }
    int i=0;
    printf("%d: %d\n", i+1 ,buff[i++]);
    printf("%d: %d\n", i+1 ,buff[i++]);
    printf("%d: %d\n", i+1 ,buff[i++]);
    printf("%d: %d\n", i+1 ,buff[i++]);
    
    int32_t final;
    
    final=0;
    final |= ( buff[i++] << 24 );
    final |= ( buff[i++] << 16 );
    final |= ( buff[i++] <<  8 );
    final |= ( buff[i++]       );
    printf("%d: %d\n", i+1 ,final);
    
    final=0;
    final |= ( buff[i++] << 24 );
    final |= ( buff[i++] << 16 );
    final |= ( buff[i++] <<  8 );
    final |= ( buff[i++]       );
    printf("%d: %d\n", i+1 ,final);
    
    final=0;
    final |= ( buff[i++] << 24 );
    final |= ( buff[i++] << 16 );
    final |= ( buff[i++] <<  8 );
    final |= ( buff[i++]       );
    printf("%d: %d\n", i+1 ,final);
    
    final=0;
    final |= ( buff[i++] << 24 );
    final |= ( buff[i++] << 16 );
    final |= ( buff[i++] <<  8 );
    final |= ( buff[i++]       );
    printf("%d: %d\n", i+1 ,final);
    
    fclose(fp_sour);
    return 0;
}